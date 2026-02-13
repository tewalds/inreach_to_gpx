#!/usr/bin/env python3
"""
Convert Garmin InReach CSV export to daily GPX files for Strava upload.

Handles:
- Filtering invalid coordinates (empty or 0)
- Automatic timezone detection from coordinates
- Splitting tracks at 2am local time
- One GPX file per day
"""

import csv
import sys
import argparse
import os
from datetime import datetime, timedelta
from collections import defaultdict
import gpxpy
import gpxpy.gpx
from timezonefinder import TimezoneFinder
import pytz

# FIT file support
try:
    from fit_tool.fit_file_builder import FitFileBuilder
    from fit_tool.profile.messages.file_id_message import FileIdMessage
    from fit_tool.profile.messages.record_message import RecordMessage
    from fit_tool.profile.messages.lap_message import LapMessage
    from fit_tool.profile.messages.session_message import SessionMessage
    from fit_tool.profile.messages.activity_message import ActivityMessage
    from fit_tool.profile.messages.event_message import EventMessage
    from fit_tool.profile.profile_type import FileType, Manufacturer, Sport, Event, EventType
    FIT_AVAILABLE = True
except ImportError:
    FIT_AVAILABLE = False


def parse_inreach_csv(csv_file):
    """Parse InReach CSV and return list of valid trackpoints."""
    trackpoints = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Extract and validate coordinates
            try:
                lat = float(row['Lat']) if row['Lat'] else 0
                lon = float(row['Lon']) if row['Lon'] else 0
                
                # Skip invalid coordinates
                if lat == 0 or lon == 0:
                    continue
                
                # Parse timestamp (UTC from GPS)
                # Example format: "3/17/2024 5:52:00 PM"
                time_str = row['GPSTime']
                utc_time = datetime.strptime(time_str, '%m/%d/%Y %I:%M:%S %p')
                utc_time = pytz.utc.localize(utc_time)
                
                # Parse altitude
                altitude = float(row['AltitudeMeters']) if row['AltitudeMeters'] else None
                
                trackpoints.append({
                    'lat': lat,
                    'lon': lon,
                    'altitude': altitude,
                    'time_utc': utc_time
                })
                
            except (ValueError, KeyError) as e:
                print(f"Skipping invalid row: {e}")
                continue
    
    return trackpoints


def convert_to_local_time(trackpoints):
    """
    Convert UTC times to local times based on coordinates.
    Optimized: only checks timezone once per UTC day using first point of that day.
    """
    if not trackpoints:
        return trackpoints
    
    # Group by UTC date to find first point of each day
    utc_days = defaultdict(list)
    for point in trackpoints:
        utc_date = point['time_utc'].date()
        utc_days[utc_date].append(point)
    
    # Get timezone for each UTC day (from first point)
    tf = TimezoneFinder()
    day_timezones = {}
    
    for utc_date, points in utc_days.items():
        first_point = points[0]
        tz_name = tf.timezone_at(lat=first_point['lat'], lng=first_point['lon'])
        day_timezones[utc_date] = tz_name if tz_name else 'UTC'
    
    # Convert all points using their day's timezone
    for point in trackpoints:
        utc_date = point['time_utc'].date()
        tz_name = day_timezones[utc_date]
        local_tz = pytz.timezone(tz_name)
        point['time_local'] = point['time_utc'].astimezone(local_tz)
        point['timezone'] = tz_name
    
    return trackpoints


def get_day_key(local_time):
    """
    Get day key for grouping tracks.
    Splits at 2am local time - times before 2am belong to previous day.
    """
    if local_time.hour < 2:
        # Before 2am - belongs to previous day
        day = local_time.date() - timedelta(days=1)
    else:
        day = local_time.date()
    
    return day


def split_by_day(trackpoints):
    """Split trackpoints into separate days (splitting at 2am local time)."""
    days = defaultdict(list)
    
    for point in trackpoints:
        day_key = get_day_key(point['time_local'])
        days[day_key].append(point)
    
    return days


def interpolate_trackpoints(trackpoints, interval_seconds, max_gap_seconds=3600, min_speed_mps=0.5):
    """
    Add interpolated points between existing trackpoints at regular time intervals.
    
    Args:
        trackpoints: List of trackpoint dictionaries
        interval_seconds: Time interval in seconds between interpolated points
        max_gap_seconds: Don't interpolate if gap is larger than this (default: 3600 = 1 hour)
        min_speed_mps: Don't interpolate if average speed is below this in m/s (default: 0.5 m/s = 1.8 km/h)
    
    Returns:
        List of trackpoints with interpolated points added
    """
    if len(trackpoints) < 2 or interval_seconds <= 0:
        return trackpoints
    
    from math import radians, sin, cos, sqrt, atan2
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate distance between two points in meters."""
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return 6371000 * c  # Earth radius in meters
    
    interpolated = []
    
    for i in range(len(trackpoints) - 1):
        current = trackpoints[i]
        next_point = trackpoints[i + 1]
        
        # Always add the current point
        interpolated.append(current)
        
        # Calculate time difference
        time_diff = (next_point['time_utc'] - current['time_utc']).total_seconds()
        
        # Skip interpolation if points are close together
        if time_diff <= interval_seconds:
            continue
        
        # Skip interpolation if gap is too large (overnight, long breaks)
        if time_diff > max_gap_seconds:
            continue
        
        # Calculate distance and speed
        distance = haversine_distance(current['lat'], current['lon'], 
                                     next_point['lat'], next_point['lon'])
        speed_mps = distance / time_diff if time_diff > 0 else 0
        
        # Skip interpolation if moving too slowly (lunch break, etc)
        if speed_mps < min_speed_mps:
            continue
        
        # Calculate number of intermediate points needed
        num_interpolated = int(time_diff / interval_seconds)
        
        # Add interpolated points
        for j in range(1, num_interpolated):
            fraction = (j * interval_seconds) / time_diff
            
            # Linear interpolation for lat, lon, altitude
            interp_point = {
                'lat': current['lat'] + fraction * (next_point['lat'] - current['lat']),
                'lon': current['lon'] + fraction * (next_point['lon'] - current['lon']),
                'time_utc': current['time_utc'] + timedelta(seconds=j * interval_seconds),
            }
            
            # Interpolate altitude if both points have it
            if current['altitude'] is not None and next_point['altitude'] is not None:
                interp_point['altitude'] = current['altitude'] + fraction * (next_point['altitude'] - current['altitude'])
            else:
                interp_point['altitude'] = current['altitude'] or next_point['altitude']
            
            # Copy timezone info from current point
            interp_point['time_local'] = interp_point['time_utc'].astimezone(pytz.timezone(current['timezone']))
            interp_point['timezone'] = current['timezone']
            
            interpolated.append(interp_point)
    
    # Add the last point
    interpolated.append(trackpoints[-1])
    
    return interpolated



def create_gpx(trackpoints, day, activity_type='hiking', trip_name=None, description=None):
    """Create GPX object from trackpoints for a given day."""
    gpx = gpxpy.gpx.GPX()
    
    # Set creator to claim barometric altimeter so Strava uses elevation data
    gpx.creator = "Garmin InReach with Barometer"
    
    # Determine the full name
    if trip_name:
        full_name = f"{trip_name} - {day.strftime('%Y-%m-%d')}"
    else:
        full_name = day.strftime('%Y-%m-%d')
    
    # Add metadata - try setting name here too
    gpx.name = full_name
    if description:
        gpx.description = description
    
    # Create track
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx_track.name = full_name
    gpx_track.type = activity_type
    gpx.tracks.append(gpx_track)
    
    # Create segment
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)
    
    # Add points
    for point in trackpoints:
        gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(
            latitude=point['lat'],
            longitude=point['lon'],
            elevation=point['altitude'],
            time=point['time_utc']  # GPX uses UTC
        ))
    
    return gpx


def create_fit(trackpoints, day, activity_type='hiking', trip_name=None):
    """Create FIT file from trackpoints for a given day."""
    if not FIT_AVAILABLE:
        raise ImportError("fit-tool library not installed. Install with: pip install fit-tool")
    
    # Map activity types to FIT Sport enum
    sport_map = {
        'hiking': Sport.HIKING,
        'biking': Sport.CYCLING,
        'running': Sport.RUNNING,
        'walking': Sport.WALKING
    }
    sport = sport_map.get(activity_type, Sport.HIKING)
    
    builder = FitFileBuilder(auto_define=True)
    
    # File ID message
    file_id = FileIdMessage()
    file_id.type = FileType.ACTIVITY
    file_id.manufacturer = Manufacturer.GARMIN.value
    file_id.product = 0
    file_id.time_created = int(trackpoints[0]['time_utc'].timestamp() * 1000)
    file_id.serial_number = 0x12345678
    builder.add(file_id)
    
    # Start event
    start_event = EventMessage()
    start_event.event = Event.TIMER
    start_event.event_type = EventType.START
    start_event.timestamp = int(trackpoints[0]['time_utc'].timestamp() * 1000)
    builder.add(start_event)
    
    # Add all record messages (trackpoints)
    for point in trackpoints:
        record = RecordMessage()
        # Try passing degrees directly - fit-tool may handle conversion
        record.position_lat = point['lat']
        record.position_long = point['lon']
        if point['altitude']:
            record.altitude = point['altitude']
        record.timestamp = int(point['time_utc'].timestamp() * 1000)
        builder.add(record)
    
    # Stop event
    stop_event = EventMessage()
    stop_event.event = Event.TIMER
    stop_event.event_type = EventType.STOP_ALL
    stop_event.timestamp = int(trackpoints[-1]['time_utc'].timestamp() * 1000)
    builder.add(stop_event)
    
    # Calculate total time and distance
    start_time = trackpoints[0]['time_utc']
    end_time = trackpoints[-1]['time_utc']
    total_elapsed_time = (end_time - start_time).total_seconds()
    
    # Calculate distance using same method as GPX
    total_distance = 0
    for i in range(1, len(trackpoints)):
        prev = trackpoints[i-1]
        curr = trackpoints[i]
        # Simple haversine distance calculation
        from math import radians, sin, cos, sqrt, atan2
        lat1, lon1 = radians(prev['lat']), radians(prev['lon'])
        lat2, lon2 = radians(curr['lat']), radians(curr['lon'])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance = 6371000 * c  # Earth radius in meters
        total_distance += distance
    
    # Lap message
    lap = LapMessage()
    lap.timestamp = int(end_time.timestamp() * 1000)
    lap.start_time = int(start_time.timestamp() * 1000)
    lap.total_elapsed_time = total_elapsed_time
    lap.total_timer_time = total_elapsed_time  # Use elapsed time as timer time
    lap.total_distance = total_distance
    lap.sport = sport
    builder.add(lap)
    
    # Session message
    session = SessionMessage()
    session.timestamp = int(end_time.timestamp() * 1000)
    session.start_time = int(start_time.timestamp() * 1000)
    session.total_elapsed_time = total_elapsed_time
    session.total_timer_time = total_elapsed_time  # Use elapsed time as timer time
    session.total_distance = total_distance
    session.sport = sport
    builder.add(session)
    
    # Activity message
    activity = ActivityMessage()
    activity.timestamp = int(end_time.timestamp() * 1000)
    activity.total_timer_time = total_elapsed_time  # Use elapsed time as timer time
    activity.num_sessions = 1
    builder.add(activity)
    
    return builder.build()


def main():
    parser = argparse.ArgumentParser(
        description='Convert Garmin InReach CSV to daily GPX files for Strava'
    )
    parser.add_argument('csv_file', help='InReach CSV export file')
    parser.add_argument('-o', '--output-dir', default='.',
                        help='Output directory for GPX files (default: current directory)')
    parser.add_argument('-t', '--type', dest='activity_type',
                        choices=['hiking', 'biking', 'running', 'walking'],
                        default='hiking',
                        help='Activity type (default: hiking)')
    parser.add_argument('-m', '--min-distance', type=float, default=1.0,
                        help='Minimum distance in km to include a day (default: 1.0)')
    parser.add_argument('--start-date', type=str,
                        help='Start date for filtering (YYYY-MM-DD format)')
    parser.add_argument('--end-date', type=str,
                        help='End date for filtering (YYYY-MM-DD format)')
    parser.add_argument('-n', '--name', type=str,
                        help='Trip name to include in GPX metadata')
    parser.add_argument('-d', '--description', type=str, 
                        default='Recorded by Garmin InReach',
                        help='Description to include in GPX metadata (default: "Recorded by Garmin InReach")')
    parser.add_argument('-f', '--format', type=str,
                        choices=['gpx', 'fit', 'both'],
                        default='gpx',
                        help='Output format: gpx, fit, or both (default: gpx)')
    parser.add_argument('--interpolate', type=int, metavar='SECONDS',
                        help='Add interpolated points every N seconds (e.g., 60 for every minute)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if args.output_dir != '.':
        os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Reading {args.csv_file}...")
    trackpoints = parse_inreach_csv(args.csv_file)
    print(f"Found {len(trackpoints)} valid trackpoints")
    
    if not trackpoints:
        print("No valid trackpoints found!")
        sys.exit(1)
    
    print("Converting to local times...")
    trackpoints = convert_to_local_time(trackpoints)
    
    print("Splitting by day (2am cutoff)...")
    days = split_by_day(trackpoints)
    
    # Parse date range filters if provided
    start_date = None
    end_date = None
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    
    # Filter by date range
    if start_date or end_date:
        filtered_days = {}
        for day, points in days.items():
            if start_date and day < start_date:
                continue
            if end_date and day > end_date:
                continue
            filtered_days[day] = points
        days = filtered_days
        
        date_range_str = ""
        if start_date and end_date:
            date_range_str = f" (filtered: {start_date} to {end_date})"
        elif start_date:
            date_range_str = f" (filtered: from {start_date})"
        elif end_date:
            date_range_str = f" (filtered: until {end_date})"
        print(f"Found {len(days)} days of data{date_range_str}:")
    else:
        print(f"\nFound {len(days)} days of data:")
    
    if not days:
        print("No days found in the specified date range!")
        sys.exit(1)
    
    # Calculate distances and filter by minimum distance
    days_with_stats = []
    total_original_points = 0
    total_interpolated_points = 0
    
    for day in sorted(days.keys()):
        points = days[day]
        
        # Calculate distance before interpolation
        gpx_temp = create_gpx(points, day, args.activity_type, args.name, args.description)
        distance_km = gpx_temp.length_3d() / 1000 if gpx_temp.length_3d() else 0
        tz = points[0]['timezone'] if points else 'UTC'
        
        # Skip if below minimum distance
        skip = distance_km < args.min_distance
        
        # Interpolate if requested and not skipping
        if args.interpolate and not skip:
            original_count = len(points)
            points = interpolate_trackpoints(points, args.interpolate)
            total_original_points += original_count
            total_interpolated_points += len(points)
        
        days_with_stats.append({
            'day': day,
            'points': points,
            'num_points': len(points),
            'distance_km': distance_km,
            'timezone': tz,
            'skip': skip
        })
    
    if args.interpolate and total_original_points > 0:
        print(f"Interpolated points: {total_original_points} -> {total_interpolated_points} total (+{total_interpolated_points - total_original_points} added)")
    
    # Check FIT support if needed
    if args.format in ['fit', 'both'] and not FIT_AVAILABLE:
        print("ERROR: fit-tool library not installed!")
        print("Install with: pip install fit-tool --break-system-packages")
        sys.exit(1)
    
    # Generate GPX files
    print(f"\nProcessing {len(days_with_stats)} days:")
    exported_count = 0
    for day_info in days_with_stats:
        day = day_info['day']
        
        if day_info['skip']:
            print(f"  {day} | {day_info['timezone']:20s} | {day_info['num_points']:3d} pts | {day_info['distance_km']:6.2f} km | SKIPPED (< {args.min_distance} km)")
        else:
            files_created = []
            
            # Create GPX if requested
            if args.format in ['gpx', 'both']:
                gpx = create_gpx(day_info['points'], day, args.activity_type, args.name, args.description)
                filename = f"track_{day.strftime('%Y-%m-%d')}.gpx"
                filepath = os.path.join(args.output_dir, filename)
                
                with open(filepath, 'w') as f:
                    f.write(gpx.to_xml())
                files_created.append(filename)
            
            # Create FIT if requested
            if args.format in ['fit', 'both']:
                fit_file = create_fit(day_info['points'], day, args.activity_type, args.name)
                filename = f"track_{day.strftime('%Y-%m-%d')}.fit"
                filepath = os.path.join(args.output_dir, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(fit_file.to_bytes())
                files_created.append(filename)
            
            files_str = ', '.join(files_created)
            print(f"  {day} | {day_info['timezone']:20s} | {day_info['num_points']:3d} pts | {day_info['distance_km']:6.2f} km | {files_str}")
            exported_count += 1
    
    format_desc = args.format.upper() if args.format != 'both' else 'GPX and FIT'
    print(f"\nExported {exported_count} days as {format_desc} files to {args.output_dir}")
    print("Upload the files to Strava.")


if __name__ == '__main__':
    main()
