"""
Database Management Module
Epic 5: Database Management - Story Points 15, 16, 17

Provides tools for managing authorized personnel database with CLI interface.
"""

import os
import cv2
import shutil
from typing import List, Dict, Any, Optional
from loguru import logger
from datetime import datetime
import json
import argparse
from pathlib import Path

from face_recognition_module import FaceDatabase


class DatabaseManager:
    """
    Manager class for database operations and administration.
    """
    
    def __init__(self, face_database: FaceDatabase, authorized_faces_dir: str, 
                 capture_source: int = 0):
        """
        Initialize database manager.
        
        Args:
            face_database: FaceDatabase instance
            authorized_faces_dir: Directory for storing authorized face images
            capture_source: Camera source for live capture
        """
        self.db = face_database
        self.authorized_faces_dir = authorized_faces_dir
        self.capture_source = capture_source
        
        # Ensure directory exists
        os.makedirs(authorized_faces_dir, exist_ok=True)
    
    def add_person_from_images(self, name: str, image_paths: List[str], 
                              department: str = "", notes: str = "") -> bool:
        """
        Add a person to database from existing images.
        
        Args:
            name: Name of the person
            image_paths: List of paths to face images
            department: Optional department information
            notes: Optional notes
            
        Returns:
            bool: True if person added successfully
        """
        try:
            # Validate images exist
            valid_paths = []
            for path in image_paths:
                if os.path.exists(path):
                    valid_paths.append(path)
                else:
                    logger.warning(f"Image not found: {path}")
            
            if not valid_paths:
                logger.error(f"No valid images found for {name}")
                return False
            
            # Create person directory
            person_dir = os.path.join(self.authorized_faces_dir, name.replace(" ", "_"))
            os.makedirs(person_dir, exist_ok=True)
            
            # Copy images to person directory
            copied_paths = []
            for i, image_path in enumerate(valid_paths):
                filename = f"{name.replace(' ', '_')}_{i+1}.jpg"
                dest_path = os.path.join(person_dir, filename)
                shutil.copy2(image_path, dest_path)
                copied_paths.append(dest_path)
                logger.info(f"Copied image: {dest_path}")
            
            # Prepare metadata
            metadata = {
                'department': department,
                'notes': notes,
                'added_by': 'admin',  # Could be extended to track actual user
                'source': 'uploaded_images'
            }
            
            # Add to database
            success = self.db.add_person(name, copied_paths, metadata)
            
            if success:
                logger.info(f"Successfully added {name} with {len(copied_paths)} images")
                return True
            else:
                # Clean up copied files if database addition failed
                for path in copied_paths:
                    if os.path.exists(path):
                        os.remove(path)
                logger.error(f"Failed to add {name} to database")
                return False
                
        except Exception as e:
            logger.error(f"Error adding person {name}: {e}")
            return False
    
    def add_person_from_camera(self, name: str, num_photos: int = 5, 
                              department: str = "", notes: str = "") -> bool:
        """
        Add a person to database by capturing photos from camera.
        
        Args:
            name: Name of the person
            num_photos: Number of photos to capture
            department: Optional department information
            notes: Optional notes
            
        Returns:
            bool: True if person added successfully
        """
        try:
            logger.info(f"Starting camera capture for {name}")
            logger.info("Press SPACE to capture photo, 'q' to quit, 'r' to retake last photo")
            
            cap = cv2.VideoCapture(self.capture_source)
            if not cap.isOpened():
                logger.error("Failed to open camera")
                return False
            
            # Create person directory
            person_dir = os.path.join(self.authorized_faces_dir, name.replace(" ", "_"))
            os.makedirs(person_dir, exist_ok=True)
            
            captured_photos = []
            photo_count = 0
            
            while photo_count < num_photos:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read from camera")
                    break
                
                # Display frame with instructions
                display_frame = frame.copy()
                cv2.putText(display_frame, f"Capturing for: {name}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Photos: {photo_count}/{num_photos}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, "SPACE: capture, Q: quit, R: retake", 
                           (10, display_frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow("Database Manager - Photo Capture", display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # Space to capture
                    filename = f"{name.replace(' ', '_')}_{photo_count+1}.jpg"
                    photo_path = os.path.join(person_dir, filename)
                    
                    cv2.imwrite(photo_path, frame)
                    captured_photos.append(photo_path)
                    photo_count += 1
                    
                    logger.info(f"Captured photo {photo_count}/{num_photos}: {filename}")
                    
                    # Brief pause and feedback
                    cv2.putText(display_frame, "CAPTURED!", 
                               (display_frame.shape[1]//2 - 100, display_frame.shape[0]//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                    cv2.imshow("Database Manager - Photo Capture", display_frame)
                    cv2.waitKey(500)
                
                elif key == ord('r') and captured_photos:  # Retake last photo
                    if captured_photos:
                        last_photo = captured_photos.pop()
                        os.remove(last_photo)
                        photo_count -= 1
                        logger.info(f"Retaking photo {photo_count+1}")
                
                elif key == ord('q'):  # Quit
                    logger.info("Capture cancelled by user")
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
            if len(captured_photos) == 0:
                logger.warning("No photos captured")
                return False
            
            # Prepare metadata
            metadata = {
                'department': department,
                'notes': notes,
                'added_by': 'admin',
                'source': 'camera_capture',
                'capture_date': datetime.now().isoformat()
            }
            
            # Add to database
            success = self.db.add_person(name, captured_photos, metadata)
            
            if success:
                logger.info(f"Successfully added {name} with {len(captured_photos)} photos")
                return True
            else:
                # Clean up files if database addition failed
                for path in captured_photos:
                    if os.path.exists(path):
                        os.remove(path)
                logger.error(f"Failed to add {name} to database")
                return False
                
        except Exception as e:
            logger.error(f"Error in camera capture for {name}: {e}")
            return False
    
    def remove_person(self, name: str, remove_files: bool = True) -> bool:
        """
        Remove a person from database.
        
        Args:
            name: Name of the person to remove
            remove_files: Whether to also remove image files
            
        Returns:
            bool: True if person removed successfully
        """
        try:
            # Remove from database
            success = self.db.remove_person(name)
            
            if success and remove_files:
                # Remove image files
                person_dir = os.path.join(self.authorized_faces_dir, name.replace(" ", "_"))
                if os.path.exists(person_dir):
                    shutil.rmtree(person_dir)
                    logger.info(f"Removed image directory: {person_dir}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error removing person {name}: {e}")
            return False
    
    def update_person_info(self, name: str, new_department: str = None, 
                          new_notes: str = None) -> bool:
        """
        Update person's metadata.
        
        Args:
            name: Name of the person
            new_department: New department (if provided)
            new_notes: New notes (if provided)
            
        Returns:
            bool: True if updated successfully
        """
        try:
            metadata = {}
            if new_department is not None:
                metadata['department'] = new_department
            if new_notes is not None:
                metadata['notes'] = new_notes
            
            if metadata:
                return self.db.update_person(name, new_metadata=metadata)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating person {name}: {e}")
            return False
    
    def list_all_people(self) -> List[Dict[str, Any]]:
        """
        Get list of all people with their information.
        
        Returns:
            List of dictionaries containing person information
        """
        people = []
        
        for name in self.db.list_all_people():
            info = self.db.get_person_info(name)
            if info:
                people.append({
                    'name': name,
                    'num_encodings': len(info['encodings']),
                    'metadata': info['metadata']
                })
        
        return people
    
    def export_database_info(self, output_file: str) -> bool:
        """
        Export database information to JSON file.
        
        Args:
            output_file: Path to output JSON file
            
        Returns:
            bool: True if exported successfully
        """
        try:
            people_info = self.list_all_people()
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_people': len(people_info),
                'people': people_info
            }
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Database info exported to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting database info: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary containing database statistics
        """
        people = self.list_all_people()
        
        total_encodings = sum(p['num_encodings'] for p in people)
        departments = {}
        
        for person in people:
            dept = person['metadata'].get('department', 'Unknown')
            departments[dept] = departments.get(dept, 0) + 1
        
        return {
            'total_people': len(people),
            'total_encodings': total_encodings,
            'average_encodings_per_person': total_encodings / len(people) if people else 0,
            'departments': departments,
            'database_file_size': os.path.getsize(self.db.encodings_file) if os.path.exists(self.db.encodings_file) else 0
        }


class DatabaseCLI:
    """
    Command Line Interface for database management.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize CLI.
        
        Args:
            db_manager: DatabaseManager instance
        """
        self.manager = db_manager
    
    def run_interactive_mode(self):
        """Run interactive CLI mode."""
        logger.info("=== Facial Detection System - Database Manager ===")
        
        while True:
            print("\nAvailable commands:")
            print("1. List all people")
            print("2. Add person (from images)")
            print("3. Add person (from camera)")
            print("4. Remove person")
            print("5. Update person info")
            print("6. Show statistics")
            print("7. Export database info")
            print("8. Exit")
            
            try:
                choice = input("\nEnter your choice (1-8): ").strip()
                
                if choice == '1':
                    self._list_people()
                elif choice == '2':
                    self._add_person_from_images()
                elif choice == '3':
                    self._add_person_from_camera()
                elif choice == '4':
                    self._remove_person()
                elif choice == '5':
                    self._update_person()
                elif choice == '6':
                    self._show_statistics()
                elif choice == '7':
                    self._export_database()
                elif choice == '8':
                    print("Goodbye!")
                    break
                else:
                    print("Invalid choice. Please try again.")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error in CLI: {e}")
    
    def _list_people(self):
        """List all people in database."""
        people = self.manager.list_all_people()
        
        if not people:
            print("No people in database.")
            return
        
        print(f"\nFound {len(people)} people in database:")
        print("-" * 80)
        
        for person in people:
            metadata = person['metadata']
            print(f"Name: {person['name']}")
            print(f"  Encodings: {person['num_encodings']}")
            print(f"  Department: {metadata.get('department', 'N/A')}")
            print(f"  Added: {metadata.get('date_added', 'N/A')}")
            print(f"  Notes: {metadata.get('notes', 'N/A')}")
            print("-" * 40)
    
    def _add_person_from_images(self):
        """Add person from existing images."""
        name = input("Enter person's name: ").strip()
        if not name:
            print("Name cannot be empty.")
            return
        
        department = input("Enter department (optional): ").strip()
        notes = input("Enter notes (optional): ").strip()
        
        image_paths = []
        while True:
            path = input("Enter image path (or press Enter to finish): ").strip()
            if not path:
                break
            image_paths.append(path)
        
        if not image_paths:
            print("At least one image path is required.")
            return
        
        success = self.manager.add_person_from_images(name, image_paths, department, notes)
        if success:
            print(f"Successfully added {name} to database!")
        else:
            print(f"Failed to add {name} to database.")
    
    def _add_person_from_camera(self):
        """Add person using camera capture."""
        name = input("Enter person's name: ").strip()
        if not name:
            print("Name cannot be empty.")
            return
        
        department = input("Enter department (optional): ").strip()
        notes = input("Enter notes (optional): ").strip()
        
        try:
            num_photos = int(input("Enter number of photos to capture (default: 5): ") or "5")
        except ValueError:
            num_photos = 5
        
        success = self.manager.add_person_from_camera(name, num_photos, department, notes)
        if success:
            print(f"Successfully added {name} to database!")
        else:
            print(f"Failed to add {name} to database.")
    
    def _remove_person(self):
        """Remove person from database."""
        people = self.manager.list_all_people()
        if not people:
            print("No people in database.")
            return
        
        print("Available people:")
        for i, person in enumerate(people, 1):
            print(f"{i}. {person['name']}")
        
        try:
            choice = int(input("Enter person number to remove: ")) - 1
            if 0 <= choice < len(people):
                name = people[choice]['name']
                confirm = input(f"Are you sure you want to remove {name}? (y/N): ")
                
                if confirm.lower() == 'y':
                    success = self.manager.remove_person(name)
                    if success:
                        print(f"Successfully removed {name} from database!")
                    else:
                        print(f"Failed to remove {name} from database.")
                else:
                    print("Removal cancelled.")
            else:
                print("Invalid choice.")
        except ValueError:
            print("Invalid input.")
    
    def _update_person(self):
        """Update person information."""
        people = self.manager.list_all_people()
        if not people:
            print("No people in database.")
            return
        
        print("Available people:")
        for i, person in enumerate(people, 1):
            print(f"{i}. {person['name']}")
        
        try:
            choice = int(input("Enter person number to update: ")) - 1
            if 0 <= choice < len(people):
                name = people[choice]['name']
                
                current_info = people[choice]['metadata']
                print(f"\nCurrent info for {name}:")
                print(f"  Department: {current_info.get('department', 'N/A')}")
                print(f"  Notes: {current_info.get('notes', 'N/A')}")
                
                new_department = input("Enter new department (press Enter to keep current): ").strip()
                new_notes = input("Enter new notes (press Enter to keep current): ").strip()
                
                success = self.manager.update_person_info(
                    name,
                    new_department if new_department else None,
                    new_notes if new_notes else None
                )
                
                if success:
                    print(f"Successfully updated {name}!")
                else:
                    print(f"Failed to update {name}.")
            else:
                print("Invalid choice.")
        except ValueError:
            print("Invalid input.")
    
    def _show_statistics(self):
        """Show database statistics."""
        stats = self.manager.get_statistics()
        
        print("\n=== Database Statistics ===")
        print(f"Total people: {stats['total_people']}")
        print(f"Total encodings: {stats['total_encodings']}")
        print(f"Average encodings per person: {stats['average_encodings_per_person']:.1f}")
        print(f"Database file size: {stats['database_file_size']} bytes")
        
        if stats['departments']:
            print("\nPeople by department:")
            for dept, count in stats['departments'].items():
                print(f"  {dept}: {count}")
    
    def _export_database(self):
        """Export database information."""
        filename = input("Enter output filename (default: database_export.json): ").strip()
        if not filename:
            filename = "database_export.json"
        
        success = self.manager.export_database_info(filename)
        if success:
            print(f"Database info exported to {filename}!")
        else:
            print("Failed to export database info.")


def main():
    """Main function to run the database management CLI."""
    try:
        import yaml
        
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize face database
        authorized_faces_dir = config['database']['authorized_faces_dir']
        encodings_file = config['database']['encodings_file']
        
        # Make paths absolute
        base_dir = os.path.dirname(os.path.dirname(__file__))
        authorized_faces_dir = os.path.join(base_dir, authorized_faces_dir)
        encodings_file = os.path.join(base_dir, encodings_file)
        
        face_db = FaceDatabase(authorized_faces_dir, encodings_file)
        
        # Initialize database manager
        manager = DatabaseManager(face_db, authorized_faces_dir, 
                                config['camera']['source'])
        
        # Run CLI
        cli = DatabaseCLI(manager)
        cli.run_interactive_mode()
        
    except KeyboardInterrupt:
        print("\nExiting database management...")
    except Exception as e:
        logger.error(f"Error running database management: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
