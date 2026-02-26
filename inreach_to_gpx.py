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
import time
from collections import defaultdict
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import gpxpy
import gpxpy.gpx
import kdtree
import pytz
import requests
from dotenv import load_dotenv
from timezonefinder import TimezoneFinder

# Constants
EARTH_RADIUS_METERS = 6371000
DEGREES_TO_METERS_APPROX = 111320  # At equator, 1 degree ≈ 111.32 km

# Strava API Endpoints
STRAVA_TOKEN_URL = "https://www.strava.com/oauth/token"
STRAVA_UPLOAD_URL = "https://www.strava.com/api/v3/uploads"

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


class StravaClient:
    """Handles Strava OAuth token refreshing and file uploads."""

    def __init__(self, client_id: str, client_secret: str, refresh_token: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.access_token = None

    def _refresh_access_token(self) -> bool:
        """Exchange refresh token for a new access token."""
        print("Refreshing Strava access token...")
        payload = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'refresh_token': self.refresh_token,
            'grant_type': 'refresh_token'
        }

        try:
            response = requests.post(STRAVA_TOKEN_URL, data=payload)
            response.raise_for_status()
            data = response.json()
            self.access_token = data['access_token']
            # Optional: update refresh token if it changed
            if 'refresh_token' in data:
                self.refresh_token = data['refresh_token']
            return True
        except Exception as e:
            print(f"Error refreshing Strava token: {e}")
            return False

    def upload_activity(self, file_path: str, name: str, description: str, activity_type: str, data_type: str) -> Optional[str]:
        """Upload a file to Strava."""
        if not self.access_token and not self._refresh_access_token():
            return None

        headers = {"Authorization": f"Bearer {self.access_token}"}

        # Map our activity types to Strava's expected sport_type
        # Strava uses 'Hike', 'Ride', 'Run', 'Walk'
        sport_map = {
            'hiking': 'Hike',
            'biking': 'Ride',
            'running': 'Run',
            'walking': 'Walk'
        }
        sport_type = sport_map.get(activity_type, 'Hike')

        payload = {
            "name": name,
            "description": description,
            "sport_type": sport_type,
            "data_type": data_type
        }

        with open(file_path, 'rb') as f:
            files = {"file": f}
            try:
                response = requests.post(STRAVA_UPLOAD_URL, headers=headers, data=payload, files=files)
                response.raise_for_status()
                return response.json().get('id_str')
            except Exception as e:
                if response.status_code == 409: # Conflict - likely duplicate
                    print(f"  Skipping {os.path.basename(file_path)}: Duplicate activity detected on Strava.")
                else:
                    print(f"  Error uploading {os.path.basename(file_path)}: {e}")
                    if hasattr(response, 'text'):
                        print(f"    Details: {response.text}")
                return None

    def check_upload_status(self, upload_id: str) -> Dict[str, Any]:
        """Check the status of an upload."""
        if not self.access_token and not self._refresh_access_token():
            return {}

        headers = {"Authorization": f"Bearer {self.access_token}"}
        url = f"{STRAVA_UPLOAD_URL}/{upload_id}"

        response = None
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if response is not None:
                return {'error': str(e), 'status_code': response.status_code}
            print(f"Error checking status for {upload_id}: {e}")
            return {}


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in meters using haversine formula."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return EARTH_RADIUS_METERS * c



def parse_inreach_csv(csv_file: str) -> List[Dict[str, Any]]:
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


def convert_to_local_time(trackpoints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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


def get_day_key(local_time: datetime) -> str:
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


def split_by_day(trackpoints: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Split trackpoints into separate days (splitting at 2am local time)."""
    days = defaultdict(list)

    for point in trackpoints:
        day_key = get_day_key(point['time_local'])
        days[day_key].append(point)

    return days


def linearly_interpolate_trackpoints(trackpoints: List[Dict[str, Any]], interval_seconds: int, max_gap_seconds: int = 3600, max_speed_kmh: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Add interpolated points between existing trackpoints at regular time intervals.

    Args:
        trackpoints: List of trackpoint dictionaries
        interval_seconds: Time interval in seconds between interpolated points
        max_gap_seconds: Don't interpolate if gap is larger than this (default: 3600 = 1 hour)
        max_speed_kmh: Don't interpolate if segment speed exceeds this (e.g., car ride)

    Returns:
        List of trackpoints with interpolated points added
    """
    if len(trackpoints) < 2 or interval_seconds <= 0:
        return trackpoints

    interpolated = []

    for i in range(len(trackpoints) - 1):
        current = trackpoints[i]
        next_point = trackpoints[i + 1]

        interpolated.append(current)

        time_diff = (next_point['time_utc'] - current['time_utc']).total_seconds()

        if time_diff <= interval_seconds:
            continue

        if time_diff > max_gap_seconds:
            continue

        # Skip interpolation if speed is too high (e.g., car ride)
        if max_speed_kmh and max_speed_kmh > 0:
            dist_m = haversine(current['lat'], current['lon'], next_point['lat'], next_point['lon'])
            speed_kmh = (dist_m / time_diff) * 3.6 if time_diff > 0 else 0
            if speed_kmh > max_speed_kmh:
                continue

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

    nodes: List[Dict[str, Any]]
    edges: Dict[int, List[Tuple[int, float]]]
    merge_threshold: float
    tree: Optional[Any]

    def __init__(self, merge_threshold: float = 10.0) -> None:
        self.nodes = []  # List of {lat, lon, elevation, node_id}
        self.edges = {}  # Dict: node_id -> [(neighbor_id, distance), ...]
        self.merge_threshold = merge_threshold
        self.tree = kdtree.KDTree()
        self.metric = kdtree.GreatCircle()

    def find_or_create_node(self, lat: float, lon: float, elevation: float) -> int:
        """Find existing node within merge_threshold or create new one."""
        entry = self.tree.find_closest((lat, lon), self.metric, self.merge_threshold)
        if entry:
            return entry.value

        node_id = len(self.nodes)
        self.nodes.append({'lat': lat, 'lon': lon, 'elevation': elevation, 'node_id': node_id})
        self.edges[node_id] = []
        self.tree.insert((lat, lon), node_id)
        return node_id

    def add_edge(self, node_a: int, node_b: int, max_segment_length: Optional[float] = None) -> None:
        """
        Add bidirectional edge between two nodes, with optional densification.

        Args:
            node_a: First node ID
            node_b: Second node ID
            max_segment_length: If set, split long edges by adding intermediate nodes
        """
        if node_a == node_b:
            return

        na = self.nodes[node_a]
        nb = self.nodes[node_b]
        dist = haversine(na['lat'], na['lon'], nb['lat'], nb['lon'])

        # If edge is too long and max_segment_length is set, add intermediate nodes
        if max_segment_length and dist > max_segment_length:
            num_segments = int(dist / max_segment_length) + 1
            prev_id = node_a

            for i in range(1, num_segments):
                fraction = i / num_segments

                intermediate_id = self.find_or_create_node(
                    na['lat'] + fraction * (nb['lat'] - na['lat']),
                    na['lon'] + fraction * (nb['lon'] - na['lon']),
                    na['elevation'] + fraction * (nb['elevation'] - na['elevation'])
                )

                # Add edge from previous to intermediate (recursively, but won't densify further)
                self.add_edge(prev_id, intermediate_id, max_segment_length=None)
                prev_id = intermediate_id

            # Add final edge to node_b
            self.add_edge(prev_id, node_b, max_segment_length=None)
        else:
            # Normal edge addition
            if node_b not in [n for n, d in self.edges[node_a]]:
                self.edges[node_a].append((node_b, dist))
            if node_a not in [n for n, d in self.edges[node_b]]:
                self.edges[node_b].append((node_a, dist))

    def add_route_gpx(self, gpx_file: str, max_segment_length: Optional[float] = None) -> int:
        """
        Add a GPX track to the graph.

        Args:
            gpx_file: Path to GPX file
            max_segment_length: If set, densify long edges by adding intermediate nodes

        Returns:
            Number of original points in the GPX file
        """
        with open(gpx_file, 'r') as f:
            gpx = gpxpy.parse(f)

        original_point_count = 0

        for track in gpx.tracks:
            for segment in track.segments:
                prev_node_id = None

                for point in segment.points:
                    original_point_count += 1

                    node_id = self.find_or_create_node(
                        point.latitude,
                        point.longitude,
                        point.elevation if point.elevation else 0
                    )

                    if prev_node_id is not None and prev_node_id != node_id:
                        self.add_edge(prev_node_id, node_id, max_segment_length)

                    prev_node_id = node_id

        return original_point_count

    def find_nearest_node(self, lat: float, lon: float) -> Tuple[float, int]:
        """Find nearest node. Returns (distance_meters, node_id)."""
        if not self.nodes:
            return (float('inf'), None)

        entry = self.tree.find_closest((lat, lon), self.metric)
        return (self.metric.dist((lat, lon), entry.p), entry.value)

    def shortest_path_astar(self, start_node: int, end_node: int, max_distance_multiplier: float = 3.0) -> Optional[List[int]]:
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

    def path_distance(self, path: List[int]) -> float:
        """Calculate total distance along a path."""
        if not path or len(path) < 2:
            return 0

        total = 0
        for i in range(len(path) - 1):
            node_a = self.nodes[path[i]]
            node_b = self.nodes[path[i + 1]]
            total += haversine(node_a['lat'], node_a['lon'], node_b['lat'], node_b['lon'])

        return total

    def point_at_distance(self, path: List[int], target_distance: float) -> Optional[Dict[str, float]]:
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

    def to_gpx(self) -> gpxpy.gpx.GPX:
        """Convert the graph back to a GPX object by reconstructing paths."""
        gpx = gpxpy.gpx.GPX()
        track = gpxpy.gpx.GPXTrack()
        gpx.tracks.append(track)

        # We'll use a simple approach to minimize segments:
        # 1. Identify all nodes with degree != 2 (junctions and endpoints) as "seed" nodes.
        # 2. For each seed node, follow edges to reconstruct continuous segments.
        # 3. Handle isolated loops.

        visited_edges = set()

        def follow_path(start_id, first_neighbor_id):
            segment = gpxpy.gpx.GPXTrackSegment()
            p = self.nodes[start_id]
            segment.points.append(gpxpy.gpx.GPXTrackPoint(p['lat'], p['lon'], elevation=p['elevation']))

            curr_id = start_id
            next_id = first_neighbor_id

            while True:
                edge = tuple(sorted((curr_id, next_id)))
                if edge in visited_edges:
                    break
                visited_edges.add(edge)

                p = self.nodes[next_id]
                segment.points.append(gpxpy.gpx.GPXTrackPoint(p['lat'], p['lon'], elevation=p['elevation']))

                # If next node has degree 2, continue path
                neighbors = [n for n, d in self.edges[next_id] if n != curr_id]
                if len(self.edges[next_id]) == 2 and neighbors:
                    curr_id = next_id
                    next_id = neighbors[0]
                else:
                    break
            return segment

        # 1. Start from junctions and endpoints
        for node_id in range(len(self.nodes)):
            if len(self.edges[node_id]) != 2:
                for neighbor_id, dist in self.edges[node_id]:
                    edge = tuple(sorted((node_id, neighbor_id)))
                    if edge not in visited_edges:
                        track.segments.append(follow_path(node_id, neighbor_id))

        # 2. Catch isolated loops
        for node_id in range(len(self.nodes)):
            for neighbor_id, dist in self.edges[node_id]:
                edge = tuple(sorted((node_id, neighbor_id)))
                if edge not in visited_edges:
                    track.segments.append(follow_path(node_id, neighbor_id))

        return gpx


def match_to_route_graph(trackpoints: List[Dict[str, Any]], route_graph: RouteGraph, snap_tolerance: float, max_route_ratio: float, interval_seconds: int, max_speed_kmh: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Match trackpoints to route graph and interpolate along the route.

    Clean logic: For each segment A→B:
    1. Determine if A and B snap to route (within tolerance)
    2. Find path between snap nodes (unconditionally - handles parallel paths)
    3. Check if route is reasonable (distance ratio)
    4. Interpolate: A→A_snap (linear), A_snap→B_snap (route), B_snap→B (linear)
    """
    result = []

    for i in range(len(trackpoints) - 1):
        a = trackpoints[i]
        b = trackpoints[i + 1]

        time_diff = (b['time_utc'] - a['time_utc']).total_seconds()
        dist_linear = haversine(a['lat'], a['lon'], b['lat'], b['lon'])
        linear_speed_kmh = (dist_linear / time_diff * 3.6) if time_diff > 0 else 0

        # Skip interpolation/matching if speed is too high (e.g., car ride)
        if max_speed_kmh and linear_speed_kmh > max_speed_kmh:
            result.append(a)
            continue

        dist_a, node_a = route_graph.find_nearest_node(a['lat'], a['lon'])
        dist_b, node_b = route_graph.find_nearest_node(b['lat'], b['lon'])

        a_on_route = (dist_a < snap_tolerance)
        b_on_route = (dist_b < snap_tolerance)

        # Always try to find route path (handles crossing/parallel cases)
        path = None
        if a_on_route or b_on_route:
            path = route_graph.shortest_path_astar(node_a, node_b, max_route_ratio)

        # Check if route is reasonable
        use_route = False
        if path:
            linear_distance = haversine(a['lat'], a['lon'], b['lat'], b['lon'])
            route_distance = route_graph.path_distance(path)

            # Only count snap distances if actually off-route
            total_distance = route_distance
            if not a_on_route:
                total_distance += dist_a
            if not b_on_route:
                total_distance += dist_b

            if linear_distance > 0 and total_distance / linear_distance <= max_route_ratio:
                use_route = True

        # Fall back to pure linear if route not usable
        if not use_route:
            linear_points = linearly_interpolate_trackpoints([a, b], interval_seconds, max_speed_kmh=max_speed_kmh)
            result.extend(linear_points[:-1])
            continue

        # Use route: build segments
        a_snap_node = route_graph.nodes[node_a]
        b_snap_node = route_graph.nodes[node_b]

        time_diff = (b['time_utc'] - a['time_utc']).total_seconds()
        route_distance = route_graph.path_distance(path)

        # Recalculate total_distance for time distribution
        total_distance = route_distance
        if not a_on_route:
            total_distance += dist_a
        if not b_on_route:
            total_distance += dist_b

        # Phase 1: Linear A → A_snap (if needed)
        if not a_on_route and dist_a > 1:
            time_fraction = dist_a / total_distance if total_distance > 0 else 0
            time_to_snap = time_diff * time_fraction

            # Interpolate elevation from A to B, not using snap node elevation
            a_snap_altitude = a.get('altitude')
            if a.get('altitude') is not None and b.get('altitude') is not None:
                a_snap_altitude = a['altitude'] + time_fraction * (b['altitude'] - a['altitude'])

            a_snap_point = {
                'lat': a_snap_node['lat'],
                'lon': a_snap_node['lon'],
                'altitude': a_snap_altitude,
                'time_utc': a['time_utc'] + timedelta(seconds=time_to_snap),
                'time_local': (a['time_utc'] + timedelta(seconds=time_to_snap)).astimezone(pytz.timezone(a['timezone'])),
                'timezone': a['timezone']
            }

            linear_points = linearly_interpolate_trackpoints([a, a_snap_point], interval_seconds)
            result.extend(linear_points[:-1])

        # Phase 2: Route A_snap → B_snap
        time_route_start = a['time_utc']
        if not a_on_route and dist_a > 1:
            time_route_start = a['time_utc'] + timedelta(seconds=time_diff * (dist_a / total_distance))

        time_route_end = b['time_utc']
        if not b_on_route and dist_b > 1:
            time_route_end = b['time_utc'] - timedelta(seconds=time_diff * (dist_b / total_distance))

        time_on_route = (time_route_end - time_route_start).total_seconds()

        if time_on_route > 0:
            for t in range(0, int(time_on_route), interval_seconds):
                fraction = t / time_on_route if time_on_route > 0 else 0
                target_dist = fraction * route_distance

                point_on_route = route_graph.point_at_distance(path, target_dist)

                # Always interpolate elevation from A to B across entire segment
                if a.get('altitude') is not None and b.get('altitude') is not None:
                    overall_time = (time_route_start + timedelta(seconds=t) - a['time_utc']).total_seconds()
                    overall_fraction = overall_time / time_diff if time_diff > 0 else 0
                    altitude = a['altitude'] + overall_fraction * (b['altitude'] - a['altitude'])
                elif point_on_route['elevation'] != 0:
                    # Fall back to route elevation if no GPS altitude
                    altitude = point_on_route['elevation']
                else:
                    altitude = a.get('altitude') or b.get('altitude')

                interp = {
                    'lat': point_on_route['lat'],
                    'lon': point_on_route['lon'],
                    'altitude': altitude,
                    'time_utc': time_route_start + timedelta(seconds=t),
                    'time_local': (time_route_start + timedelta(seconds=t)).astimezone(pytz.timezone(a['timezone'])),
                    'timezone': a['timezone']
                }
                result.append(interp)

        # Phase 3: Linear B_snap → B (if needed)
        if not b_on_route and dist_b > 1:
            time_fraction = dist_b / total_distance if total_distance > 0 else 0
            time_from_snap = time_diff * time_fraction

            # Calculate time fraction for B_snap in overall A→B segment
            b_snap_time_fraction = (time_diff - time_from_snap) / time_diff if time_diff > 0 else 1.0

            # Interpolate elevation from A to B, not using snap node elevation
            b_snap_altitude = b.get('altitude')
            if a.get('altitude') is not None and b.get('altitude') is not None:
                b_snap_altitude = a['altitude'] + b_snap_time_fraction * (b['altitude'] - a['altitude'])

            b_snap_point = {
                'lat': b_snap_node['lat'],
                'lon': b_snap_node['lon'],
                'altitude': b_snap_altitude,
                'time_utc': b['time_utc'] - timedelta(seconds=time_from_snap),
                'time_local': (b['time_utc'] - timedelta(seconds=time_from_snap)).astimezone(pytz.timezone(b['timezone'])),
                'timezone': b['timezone']
            }

            linear_points = linearly_interpolate_trackpoints([b_snap_point, b], interval_seconds)
            result.extend(linear_points[:-1])

    result.append(trackpoints[-1])
    return result


def create_gpx(trackpoints: List[Dict[str, Any]], day: str, activity_type: str = 'hiking', trip_name: Optional[str] = None, description: Optional[str] = None) -> gpxpy.gpx.GPX:
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


def create_fit(trackpoints: List[Dict[str, Any]], day: str, activity_type: str = 'hiking', trip_name: Optional[str] = None) -> bytes:
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


def main() -> None:
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
    parser.add_argument('--max-speed', type=float, default=20.0,
                        help='Maximum speed in km/h to include a point (default: 20.0)')
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
    parser.add_argument('--route-tolerance', type=float, default=200.0,
                        help='Maximum distance in meters to snap to route (default: 200)')
    parser.add_argument('--route-merge', type=float, default=10.0,
                        help='Distance in meters to merge route nodes (default: 10)')
    parser.add_argument('--max-route-ratio', type=float, default=3.0,
                        help='Maximum ratio of route_distance/linear_distance (default: 3.0)')
    parser.add_argument('--export-route-gpx', type=str,
                        help='Export the deduped route graph to a new GPX file and exit')

    # Strava Upload Options
    parser.add_argument('--strava-upload', action='store_true',
                        help='Automatically upload generated files to Strava (requires .env configuration)')
    parser.add_argument('--strava-client-id', type=str,
                        help='Strava API Client ID')
    parser.add_argument('--strava-client-secret', type=str,
                        help='Strava API Client Secret')
    parser.add_argument('--strava-refresh-token', type=str,
                        help='Strava API Refresh Token')

    args = parser.parse_args()

    # Load environment variables if they exist
    load_dotenv()

    # Priority: CLI arguments -> environment variables
    strava_client_id = args.strava_client_id or os.getenv('STRAVA_CLIENT_ID')
    strava_client_secret = args.strava_client_secret or os.getenv('STRAVA_CLIENT_SECRET')
    strava_refresh_token = args.strava_refresh_token or os.getenv('STRAVA_REFRESH_TOKEN')

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
        # Resolve any directories into files
        resolved_files = []
        for path in args.route_gpx_files:
            if os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.lower().endswith('.gpx'):
                            resolved_files.append(os.path.join(root, file))
            elif os.path.isfile(path):
                resolved_files.append(path)
            else:
                print(f"Warning: Path not found: {path}")

        if not resolved_files:
            print("Warning: No GPX files found in specified routes!")
        else:
            print(f"\nLoading {len(resolved_files)} route GPX file(s)...")
            route_graph = RouteGraph(merge_threshold=args.route_merge)

            max_segment_length = args.route_tolerance / 4
            print(f"  Densifying long edges (>{max_segment_length:.0f}m) during load...")

            total_original_points = 0
            for gpx_file in sorted(resolved_files):
                print(f"  Loading {gpx_file}...")
                original_points = route_graph.add_route_gpx(gpx_file, max_segment_length=max_segment_length)
                total_original_points += original_points
                print(f"    {original_points} original points")

            print(f"  Route graph: {len(route_graph.nodes)} nodes ({total_original_points} original), {sum(len(edges) for edges in route_graph.edges.values())} edges")

            if args.export_route_gpx:
                print(f"\nExporting route graph to {args.export_route_gpx}...")
                export_gpx = route_graph.to_gpx()
                with open(args.export_route_gpx, 'w') as f:
                    f.write(export_gpx.to_xml())
                print("Done. You can now use this file with --route-gpx for faster future loads.")
                sys.exit(0)

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
                points = match_to_route_graph(
                    points,
                    route_graph,
                    args.route_tolerance,
                    args.max_route_ratio,
                    args.interpolate if args.interpolate else 10,
                    max_speed_kmh=args.max_speed
                )
            elif args.interpolate:
                points = linearly_interpolate_trackpoints(
                    points,
                    args.interpolate,
                    max_speed_kmh=args.max_speed
                )

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

    # Initialize Strava client if requested
    strava_client = None
    if args.strava_upload and strava_client_id and strava_client_secret and strava_refresh_token:
        strava_client = StravaClient(strava_client_id, strava_client_secret, strava_refresh_token)

    pending_uploads = []

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

            # Upload to Strava if client is ready
            if strava_client:
                # Upload the best available format (FIT > GPX)
                upload_file = None
                upload_format = None
                if 'track_' + day.strftime('%Y-%m-%d') + '.fit' in files_created:
                    upload_file = os.path.join(args.output_dir, 'track_' + day.strftime('%Y-%m-%d') + '.fit')
                    upload_format = 'fit'
                elif 'track_' + day.strftime('%Y-%m-%d') + '.gpx' in files_created:
                    upload_file = os.path.join(args.output_dir, 'track_' + day.strftime('%Y-%m-%d') + '.gpx')
                    upload_format = 'gpx'

                if upload_file:
                    print(f"    Queueing for Strava upload...", end="", flush=True)
                    upload_name = args.name + " - " + day.strftime('%Y-%m-%d') if args.name else day.strftime('%Y-%m-%d')
                    upload_id = strava_client.upload_activity(
                        upload_file,
                        upload_name,
                        args.description,
                        args.activity_type,
                        upload_format
                    )
                    if upload_id:
                        print(f" DONE (ID: {upload_id})")
                        pending_uploads.append({'day': day, 'upload_id': upload_id, 'name': upload_name})
                    else:
                        print(" FAILED")

    # Poll for Strava activity IDs and links
    if pending_uploads:
        print(f"\nWaiting for Strava to process {len(pending_uploads)} upload(s)...")
        results = []
        still_pending = pending_uploads

        for attempt in range(1, 10):
            print(f"  {len(still_pending)} processing... checking in {attempt * 60}s")
            time.sleep(attempt * 60)

            next_still_pending = []
            rate_limited = False
            for upload in still_pending:
                status = strava_client.check_upload_status(upload['upload_id'])
                if status.get('status_code') == 429:
                    print(f"  Rate limit exceeded (429). Skipping further status checks.")
                    rate_limited = True
                    break

                if status.get('status') == 'Your upload is ready.':
                    activity_id = status.get('activity_id')
                    results.append({
                        'day': upload['day'],
                        'name': upload['name'],
                        'link': f"https://www.strava.com/activities/{activity_id}"
                    })
                elif status.get('error'):
                    print(f"  Error with upload for {upload['day']}: {status.get('error')}")
                else:
                    # Still processing
                    next_still_pending.append(upload)

            if rate_limited:
                break

            still_pending = next_still_pending
            if not still_pending:
                break

        if results:
            print("\nStrava Activity Links:")
            for res in sorted(results, key=lambda x: x['day']):
                print(f"  {res['day']}: {res['link']} ({res['name']})")

        if still_pending:
            print(f"\n{len(still_pending)} upload(s) are still processing. Check your Strava dashboard later.")

    format_desc = args.format.upper() if args.format != 'both' else 'GPX and FIT'
    print(f"\nExported {exported_count} days as {format_desc} files to {args.output_dir}")


if __name__ == '__main__':
    main()
