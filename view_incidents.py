#!/usr/bin/env python3
"""View and analyze incident reports."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path


def format_timestamp(iso_str: str) -> str:
    """Format ISO timestamp for display."""
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return iso_str


def display_incident_report(report_path: str, show_motion: bool = False, show_damage: bool = True):
    """Display a formatted incident report."""
    if not os.path.exists(report_path):
        print(f"Report not found: {report_path}")
        return
    
    with open(report_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    summary = data.get('summary', {})
    
    print("\n" + "="*80)
    print(f"INCIDENT REPORT: {summary.get('video_file', 'Unknown')}")
    print("="*80)
    
    print(f"\nProcessing Time: {format_timestamp(summary.get('start_time', ''))} to {format_timestamp(summary.get('end_time', ''))}")
    print(f"Total Frames: {summary.get('total_frames', 0)}")
    print(f"Output Video: {summary.get('output_video', 'N/A')}")
    
    print(f"\nIncident Summary:")
    print(f"  Motion/Intrusion Events: {summary.get('motion_incidents', 0)}")
    print(f"  Damage Detection Events: {summary.get('damage_incidents', 0)}")
    
    # Configuration
    config = summary.get('configuration', {})
    print(f"\nConfiguration:")
    print(f"  Person-Only Mode: {config.get('person_only_mode', False)}")
    print(f"  Damage Detection: {config.get('damage_detection_enabled', False)}")
    if config.get('damage_detection_enabled'):
        print(f"  - Diff Threshold: {config.get('damage_diff_threshold', 'N/A')}")
        print(f"  - Edge Spike Ratio: {config.get('damage_edge_spike_ratio', 'N/A')}")
        print(f"  - Fragment Threshold: {config.get('damage_fragment_threshold', 'N/A')}")
        print(f"  - Persistence: {config.get('damage_persistence', 'N/A')}")
    
    # Statistics
    stats = data.get('statistics', {})
    print(f"\nStatistics:")
    print(f"  Processing Duration: {stats.get('processing_duration_seconds', 0):.2f}s")
    print(f"  Motion Detection Rate: {stats.get('motion_detection_rate', 0):.4f}")
    print(f"  Total Motion Frames: {stats.get('total_motion_frames', 0)}")
    print(f"  Unique Damage ROIs: {stats.get('unique_damage_rois', 0)}")
    if stats.get('first_intrusion_frame') is not None:
        print(f"  First Intrusion: Frame {stats.get('first_intrusion_frame')}")
        print(f"  Last Intrusion: Frame {stats.get('last_intrusion_frame')}")
    
    # Motion summary (compact)
    motion_summary = data.get('motion_summary', {})
    if motion_summary and motion_summary.get('intrusion_periods'):
        periods = motion_summary['intrusion_periods']
        print(f"\n{'-'*80}")
        print(f"INTRUSION PERIODS ({motion_summary.get('total_intrusion_periods', 0)} periods)")
        print(f"{'-'*80}")
        print(f"Max Consecutive Frames: {motion_summary.get('max_consecutive_frames', 0)}")
        print(f"\nPeriod Details:")
        for i, period in enumerate(periods[:10], 1):  # Show first 10
            print(f"  [{i}] Frames {period['start_frame']}-{period['end_frame']} "
                  f"({period['duration_frames']} frames, max {period['max_persons_detected']} person(s))")
        if len(periods) > 10:
            print(f"  ... and {len(periods) - 10} more periods")
    
    # Damage incidents (detailed)
    if show_damage:
        damage_incidents = data.get('damage_incidents', [])
        if damage_incidents:
            print(f"\n{'-'*80}")
            print(f"DAMAGE EVENTS ({len(damage_incidents)} total)")
            print(f"{'-'*80}")
            for i, incident in enumerate(damage_incidents, 1):
                print(f"\n[{i}] {incident.get('event_type', 'unknown').upper()}")
                print(f"    Time: {format_timestamp(incident.get('timestamp', ''))}")
                print(f"    Frame: {incident.get('frame_number', 'N/A')}")
                print(f"    ROI: {incident.get('roi_type', 'unknown')} #{incident.get('roi_index', 0)}")
                print(f"    Score: {incident.get('score', 0):.2f}")
                print(f"    Location: {incident.get('roi_coordinates', [])}")
    
    # Motion incidents (optional, can be many)
    if show_motion:
        motion_incidents = data.get('motion_incidents', [])
        if motion_incidents:
            print(f"\n{'-'*80}")
            print(f"MOTION/INTRUSION EVENTS ({len(motion_incidents)} total)")
            print(f"{'-'*80}")
            # Show first 10 and last 10
            to_show = motion_incidents[:10] + (motion_incidents[-10:] if len(motion_incidents) > 20 else [])
            for i, incident in enumerate(to_show, 1):
                if i == 11 and len(motion_incidents) > 20:
                    print(f"\n... ({len(motion_incidents) - 20} events omitted) ...\n")
                print(f"[Frame {incident.get('frame_number', 'N/A')}] {incident.get('event_type', 'unknown')} - "
                      f"{incident.get('person_count', 0)} person(s)")
    
    print("\n" + "="*80 + "\n")


def display_consolidated_report(report_path: str):
    """Display a consolidated summary report."""
    if not os.path.exists(report_path):
        print(f"Report not found: {report_path}")
        return
    
    with open(report_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("\n" + "="*80)
    print("CONSOLIDATED INCIDENT REPORT")
    print("="*80)
    
    print(f"\nGenerated: {format_timestamp(data.get('generated_at', ''))}")
    print(f"Videos Processed: {data.get('total_videos_processed', 0)}")
    print(f"Total Frames: {data.get('total_frames_processed', 0)}")
    print(f"Total Motion Incidents: {data.get('total_motion_incidents', 0)}")
    print(f"Total Damage Incidents: {data.get('total_damage_incidents', 0)}")
    
    # Per-video summaries
    print(f"\n{'-'*80}")
    print("PER-VIDEO SUMMARY")
    print(f"{'-'*80}")
    for summary in data.get('video_summaries', []):
        print(f"\n{summary.get('video_file', 'Unknown')}:")
        print(f"  Frames: {summary.get('total_frames', 0)}")
        print(f"  Motion Events: {summary.get('motion_incidents', 0)}")
        print(f"  Damage Events: {summary.get('damage_incidents', 0)}")
    
    # All damage events
    all_damage = data.get('all_damage_events', [])
    if all_damage:
        print(f"\n{'-'*80}")
        print(f"ALL DAMAGE EVENTS ({len(all_damage)} total)")
        print(f"{'-'*80}")
        for i, event in enumerate(all_damage, 1):
            print(f"\n[{i}] {event.get('video_file', 'Unknown')} - Frame {event.get('frame_number', 'N/A')}")
            print(f"    {event.get('roi_type', 'unknown')} {event.get('event_type', 'unknown')} (score: {event.get('score', 0):.2f})")
    
    print("\n" + "="*80 + "\n")


def list_incident_reports(incidents_dir: str = './data/incidents/'):
    """List all available incident reports."""
    if not os.path.exists(incidents_dir):
        print(f"Incidents directory not found: {incidents_dir}")
        return
    
    files = [f for f in os.listdir(incidents_dir) if f.endswith('.json')]
    
    if not files:
        print(f"No incident reports found in {incidents_dir}")
        return
    
    print(f"\nAvailable incident reports in {incidents_dir}:")
    print("-" * 80)
    
    consolidated = [f for f in files if f.startswith('consolidated_report_')]
    individual = [f for f in files if not f.startswith('consolidated_report_')]
    
    if consolidated:
        print("\nConsolidated Reports:")
        for f in sorted(consolidated):
            path = os.path.join(incidents_dir, f)
            size = os.path.getsize(path)
            mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
            print(f"  {f} ({size} bytes, modified: {mtime})")
    
    if individual:
        print("\nIndividual Video Reports:")
        for f in sorted(individual):
            path = os.path.join(incidents_dir, f)
            size = os.path.getsize(path)
            mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
            print(f"  {f} ({size} bytes, modified: {mtime})")
    
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View incident reports")
    parser.add_argument("--list", action="store_true", help="List all available reports")
    parser.add_argument("--file", type=str, help="Path to specific incident report JSON file")
    parser.add_argument("--consolidated", action="store_true", help="Display consolidated report (latest)")
    parser.add_argument("--show-motion", action="store_true", help="Show motion events in detail")
    parser.add_argument("--incidents-dir", default="./data/incidents/", help="Incidents directory")
    
    args = parser.parse_args()
    
    if args.list:
        list_incident_reports(args.incidents_dir)
    elif args.file:
        if 'consolidated' in args.file:
            display_consolidated_report(args.file)
        else:
            display_incident_report(args.file, show_motion=args.show_motion)
    elif args.consolidated:
        # Find latest consolidated report
        incidents_dir = args.incidents_dir
        if os.path.exists(incidents_dir):
            files = [
                os.path.join(incidents_dir, f)
                for f in os.listdir(incidents_dir)
                if f.startswith('consolidated_report_') and f.endswith('.json')
            ]
            if files:
                latest = max(files, key=os.path.getmtime)
                display_consolidated_report(latest)
            else:
                print("No consolidated reports found")
        else:
            print(f"Incidents directory not found: {incidents_dir}")
    else:
        # Show latest individual report
        incidents_dir = args.incidents_dir
        if os.path.exists(incidents_dir):
            files = [
                os.path.join(incidents_dir, f)
                for f in os.listdir(incidents_dir)
                if not f.startswith('consolidated_report_') and f.endswith('.json')
            ]
            if files:
                latest = max(files, key=os.path.getmtime)
                print(f"Showing latest report: {os.path.basename(latest)}")
                display_incident_report(latest, show_motion=args.show_motion)
            else:
                print("No incident reports found. Use --list to see all options.")
        else:
            print(f"Incidents directory not found: {incidents_dir}")
            print("Run the intrusion detection system first to generate reports.")
