#!/usr/bin/env python3
"""
Convert Garmin InReach CSV export to daily GPX files for Strava upload.

Handles:
- Filtering invalid coordinates (empty or 0)
- Automatic timezone detection from coordinates
- Splitting tracks at 2am local time
- One GPX file per day
"""

# Standard library imports
import argparse
import csv
import heapq
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2

# Third-party imports
import gpxpy
import gpxpy.gpx
import pytz
from timezonefinder import TimezoneFinder

# Constants
EARTH_RADIUS_METERS = 6371000
DEGREES_TO_METERS_APPROX = 111320  # At equator, 1 degree ≈ 111.32 km

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

# Scipy for route matching
try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in meters using haversine formula."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return EARTH_RADIUS_METERS * c



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


def linearly_interpolate_trackpoints(trackpoints, interval_seconds, max_gap_seconds=3600, min_speed_mps=0.5):
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

        # Calculate distance and speed using shared haversine
        distance = haversine(current['lat'], current['lon'], next_point['lat'], next_point['lon'])
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


class RouteGraph:
    """
    Undirected graph representation of trail route(s).
    Handles multiple GPX files, trail junctions, and bidirectional paths.
    """

    def __init__(self, merge_threshold=10.0):
        self.nodes = []  # List of {lat, lon, elevation, node_id}
        self.edges = {}  # Dict: node_id -> [(neighbor_id, distance), ...]
        self.merge_threshold = merge_threshold
        self.tree = None
        self._tree_dirty = False

    def _rebuild_tree_if_needed(self):
        """Rebuild spatial index only when needed (after batch node additions)."""
        if self._tree_dirty and self.nodes:
            if not SCIPY_AVAILABLE:
                raise ImportError("scipy is required for route matching")
            coords = [(n['lat'], n['lon']) for n in self.nodes]
            self.tree = cKDTree(coords)
            self._tree_dirty = False

    def find_or_create_node(self, lat, lon, elevation):
        """Find existing node within merge_threshold or create new one."""
        if not self.nodes:
            node_id = len(self.nodes)
            self.nodes.append({'lat': lat, 'lon': lon, 'elevation': elevation, 'node_id': node_id})
            self.edges[node_id] = []
            self._tree_dirty = True
            return node_id

        # Rebuild tree if needed before querying
        self._rebuild_tree_if_needed()

        # Find nearest node - dist is in degrees
        dist_degrees, idx = self.tree.query([lat, lon])
        dist_meters = dist_degrees * DEGREES_TO_METERS_APPROX

        if dist_meters < self.merge_threshold:
            return idx
        else:
            # Create new node
            node_id = len(self.nodes)
            self.nodes.append({'lat': lat, 'lon': lon, 'elevation': elevation, 'node_id': node_id})
            self.edges[node_id] = []
            self._tree_dirty = True
            return node_id

    def add_edge(self, node_a, node_b):
        """Add bidirectional edge between two nodes."""
        if node_a == node_b:
            return  # Skip self-loops

        # Calculate distance using shared haversine
        na = self.nodes[node_a]
        nb = self.nodes[node_b]
        dist = haversine(na['lat'], na['lon'], nb['lat'], nb['lon'])

        # Add bidirectional edges (if not already present)
        if node_b not in [n for n, d in self.edges[node_a]]:
            self.edges[node_a].append((node_b, dist))
        if node_a not in [n for n, d in self.edges[node_b]]:
            self.edges[node_b].append((node_a, dist))

    def add_route_gpx(self, gpx_file):
        """Add a GPX track to the graph."""
        with open(gpx_file, 'r') as f:
            gpx = gpxpy.parse(f)

        for track in gpx.tracks:
            for segment in track.segments:
                prev_node_id = None

                for point in segment.points:
                    node_id = self.find_or_create_node(
                        point.latitude,
                        point.longitude,
                        point.elevation if point.elevation else 0
                    )

                    if prev_node_id is not None and prev_node_id != node_id:
                        self.add_edge(prev_node_id, node_id)

                    prev_node_id = node_id

        # Rebuild tree once after loading entire GPX
        self._rebuild_tree_if_needed()

    def find_nearest_node(self, lat, lon):
        """Find nearest node. Returns (distance_meters, node_id)."""
        if not self.nodes:
            return (float('inf'), None)

        self._rebuild_tree_if_needed()

        # Query returns distance in degrees
        dist_degrees, idx = self.tree.query([lat, lon])
        dist_meters = dist_degrees * DEGREES_TO_METERS_APPROX
        return (dist_meters, idx)

    def shortest_path_astar(self, start_node, end_node, max_distance_multiplier=3.0):
        """
        A* shortest path between nodes with early termination.

        Args:
            start_node: Starting node ID
            end_node: Ending node ID
            max_distance_multiplier: Stop searching if g_score exceeds heuristic * this value

        Returns:
            List of node_ids or None if not connected or path too long
        """
        if start_node == end_node:
            return [start_node]

        # Heuristic: haversine distance to goal
        goal = self.nodes[end_node]
        start = self.nodes[start_node]
        straight_line_distance = haversine(start['lat'], start['lon'], goal['lat'], goal['lon'])
        max_search_distance = straight_line_distance * max_distance_multiplier

        def heuristic(node_id):
            node = self.nodes[node_id]
            return haversine(node['lat'], node['lon'], goal['lat'], goal['lon'])

        # A* search with distance cutoff
        open_set = [(heuristic(start_node), 0, start_node, [start_node])]
        visited = set()

        while open_set:
            f_score, g_score, current, path = heapq.heappop(open_set)

            if current == end_node:
                return path

            if current in visited:
                continue

            # Early termination if we've gone too far
            if g_score > max_search_distance:
                continue

            visited.add(current)

            for neighbor, edge_dist in self.edges[current]:
                if neighbor in visited:
                    continue

                new_g = g_score + edge_dist

                # Skip if this path is already too long
                if new_g > max_search_distance:
                    continue

                new_f = new_g + heuristic(neighbor)
                new_path = path + [neighbor]

                heapq.heappush(open_set, (new_f, new_g, neighbor, new_path))

        return None  # No path found within distance limit

    def path_distance(self, path):
        """Calculate total distance along a path."""
        if not path or len(path) < 2:
            return 0

        total = 0
        for i in range(len(path) - 1):
            node_a = self.nodes[path[i]]
            node_b = self.nodes[path[i + 1]]
            total += haversine(node_a['lat'], node_a['lon'], node_b['lat'], node_b['lon'])

        return total

    def point_at_distance(self, path, target_distance):
        """
        Get point at a specific distance along the path.
        Returns {lat, lon, elevation}.
        """
        if not path:
            return None

        if target_distance <= 0:
            node = self.nodes[path[0]]
            return {'lat': node['lat'], 'lon': node['lon'], 'elevation': node['elevation']}

        cumulative = 0
        for i in range(len(path) - 1):
            node_a = self.nodes[path[i]]
            node_b = self.nodes[path[i + 1]]
            segment_dist = haversine(node_a['lat'], node_a['lon'], node_b['lat'], node_b['lon'])

            if cumulative + segment_dist >= target_distance:
                # Target is in this segment
                fraction = (target_distance - cumulative) / segment_dist if segment_dist > 0 else 0

                return {
                    'lat': node_a['lat'] + fraction * (node_b['lat'] - node_a['lat']),
                    'lon': node_a['lon'] + fraction * (node_b['lon'] - node_a['lon']),
                    'elevation': node_a['elevation'] + fraction * (node_b['elevation'] - node_a['elevation'])
                }

            cumulative += segment_dist

        # Target is beyond path end
        node = self.nodes[path[-1]]
        return {'lat': node['lat'], 'lon': node['lon'], 'elevation': node['elevation']}


def match_to_route_graph(trackpoints, route_graph, snap_tolerance, max_route_ratio, interval_seconds):
    """
    Match trackpoints to route graph and interpolate along the route.

    Args:
        trackpoints: List of GPS trackpoints
        route_graph: RouteGraph object
        snap_tolerance: Max distance to snap to route (meters)
        max_route_ratio: Max ratio of route_distance/linear_distance
        interval_seconds: Time between interpolated points

    Returns:
        List of trackpoints with route-matched interpolation
    """
    result = []

    for i in range(len(trackpoints) - 1):
        current = trackpoints[i]
        next_pt = trackpoints[i + 1]

        # Find nearest route nodes
        dist_current, node_current = route_graph.find_nearest_node(current['lat'], current['lon'])
        dist_next, node_next = route_graph.find_nearest_node(next_pt['lat'], next_pt['lon'])

        # Check if both points snap to route
        on_route = (dist_current < snap_tolerance and dist_next < snap_tolerance and
                   node_current is not None and node_next is not None)

        if on_route:
            # Find path on route (with distance cutoff based on max_route_ratio)
            path = route_graph.shortest_path_astar(node_current, node_next, max_route_ratio)

            if path:
                route_distance = route_graph.path_distance(path)
                linear_distance = haversine(current['lat'], current['lon'], next_pt['lat'], next_pt['lon'])

                # Sanity check (A* already filtered, but double-check)
                if linear_distance > 0 and route_distance / linear_distance <= max_route_ratio:
                    # Good match - interpolate along route
                    time_diff = (next_pt['time_utc'] - current['time_utc']).total_seconds()

                    # Add interpolated points along the route
                    for t in range(0, int(time_diff), interval_seconds):
                        fraction = t / time_diff if time_diff > 0 else 0
                        target_dist = fraction * route_distance

                        point_on_route = route_graph.point_at_distance(path, target_dist)

                        interp = {
                            'lat': point_on_route['lat'],
                            'lon': point_on_route['lon'],
                            'altitude': point_on_route['elevation'],
                            'time_utc': current['time_utc'] + timedelta(seconds=t),
                            'time_local': (current['time_utc'] + timedelta(seconds=t)).astimezone(
                                pytz.timezone(current['timezone'])
                            ),
                            'timezone': current['timezone']
                        }
                        result.append(interp)

                    continue  # Skip linear interpolation

        # Off-route or failed sanity check - use linear interpolation
        result.extend(linearly_interpolate_trackpoints([current, next_pt], interval_seconds))

    # Add final point
    result.append(trackpoints[-1])

    return result



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

    # Calculate distance using shared haversine function
    total_distance = 0
    for i in range(1, len(trackpoints)):
        prev = trackpoints[i-1]
        curr = trackpoints[i]
        total_distance += haversine(prev['lat'], prev['lon'], curr['lat'], curr['lon'])

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
                        help='Add interpolated points every N seconds (e.g., 10 for best Strava results)')
    parser.add_argument('--route-gpx', action='append', dest='route_gpx_files',
                        help='Route GPX file(s) to match against. Can be specified multiple times for multiple routes.')
    parser.add_argument('--route-tolerance', type=float, default=100.0,
                        help='Maximum distance in meters to snap to route (default: 100)')
    parser.add_argument('--route-merge', type=float, default=10.0,
                        help='Distance in meters to merge route nodes (default: 10)')
    parser.add_argument('--max-route-ratio', type=float, default=3.0,
                        help='Maximum ratio of route_distance/linear_distance (default: 3.0)')

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

    # Load route GPX files if provided
    route_graph = None
    if args.route_gpx_files:
        if not SCIPY_AVAILABLE:
            print("ERROR: scipy is required for route matching!")
            print("Install with: pip install scipy --break-system-packages")
            sys.exit(1)

        print(f"\nLoading {len(args.route_gpx_files)} route GPX file(s)...")
        route_graph = RouteGraph(merge_threshold=args.route_merge)

        for gpx_file in args.route_gpx_files:
            print(f"  Loading {gpx_file}...")
            route_graph.add_route_gpx(gpx_file)

        print(f"  Route graph: {len(route_graph.nodes)} nodes, {sum(len(edges) for edges in route_graph.edges.values())} edges")

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
        if (args.interpolate or route_graph) and not skip:
            original_count = len(points)

            if route_graph:
                # Use route-matched interpolation
                points = match_to_route_graph(
                    points,
                    route_graph,
                    args.route_tolerance,
                    args.max_route_ratio,
                    args.interpolate if args.interpolate else 10
                )
            elif args.interpolate:
                # Use linear interpolation
                points = linearly_interpolate_trackpoints(points, args.interpolate)

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