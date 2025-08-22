"""
Voice Clone Database Manager
Handles SQLite database operations for voice clones with categories and search functionality
"""

import sqlite3
import json
import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

class VoiceCloneDatabase:
    def __init__(self, db_path: str = "voice_clones.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the voice clones database with proper schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create voice_clones table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS voice_clones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                voice_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                category TEXT DEFAULT 'Personal',
                description TEXT,
                tags TEXT,  -- JSON array of tags
                created_date TEXT NOT NULL,
                last_used TEXT,
                usage_count INTEGER DEFAULT 0,
                quality_rating INTEGER DEFAULT 0,  -- 1-5 stars
                is_favorite BOOLEAN DEFAULT 0,
                metadata TEXT  -- JSON for additional data
            )
        """)
        
        # Create categories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                color TEXT DEFAULT '#3498db',
                description TEXT,
                created_date TEXT NOT NULL
            )
        """)
        
        # Insert default categories if they don't exist
        default_categories = [
            ("Personal", "#3498db", "Personal voice clones"),
            ("Professional", "#2ecc71", "Business and professional voices"),
            ("Characters", "#9b59b6", "Character voices and personas"),
            ("Languages", "#e74c3c", "Different language voices"),
            ("Experimental", "#f39c12", "Testing and experimental voices")
        ]
        
        for cat_name, color, desc in default_categories:
            cursor.execute("""
                INSERT OR IGNORE INTO categories (name, color, description, created_date)
                VALUES (?, ?, ?, ?)
            """, (cat_name, color, desc, datetime.datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def add_voice_clone(self, voice_id: str, name: str, category: str = "Personal", 
                       description: str = "", tags: List[str] = None, 
                       quality_rating: int = 0, metadata: Dict = None) -> bool:
        """Add a new voice clone to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            tags_json = json.dumps(tags or [])
            metadata_json = json.dumps(metadata or {})
            created_date = datetime.datetime.now().isoformat()
            
            cursor.execute("""
                INSERT INTO voice_clones 
                (voice_id, name, category, description, tags, created_date, quality_rating, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (voice_id, name, category, description, tags_json, created_date, quality_rating, metadata_json))
            
            conn.commit()
            conn.close()
            return True
            
        except sqlite3.IntegrityError:
            # Voice ID already exists
            return False
        except Exception as e:
            print(f"Error adding voice clone: {e}")
            return False
    
    def get_voice_clones(self, category: str = None, search_term: str = None, 
                        favorites_only: bool = False, limit: int = None) -> List[Dict]:
        """Get voice clones with optional filtering"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT voice_id, name, category, description, tags, created_date, 
                   last_used, usage_count, quality_rating, is_favorite, metadata
            FROM voice_clones 
            WHERE 1=1
        """
        params = []
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        if search_term:
            query += " AND (name LIKE ? OR description LIKE ? OR tags LIKE ?)"
            search_pattern = f"%{search_term}%"
            params.extend([search_pattern, search_pattern, search_pattern])
        
        if favorites_only:
            query += " AND is_favorite = 1"
        
        query += " ORDER BY usage_count DESC, created_date DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        voice_clones = []
        for row in rows:
            voice_clone = {
                'voice_id': row[0],
                'name': row[1],
                'category': row[2],
                'description': row[3],
                'tags': json.loads(row[4]) if row[4] else [],
                'created_date': row[5],
                'last_used': row[6],
                'usage_count': row[7],
                'quality_rating': row[8],
                'is_favorite': bool(row[9]),
                'metadata': json.loads(row[10]) if row[10] else {}
            }
            voice_clones.append(voice_clone)
        
        conn.close()
        return voice_clones
    
    def update_voice_usage(self, voice_id: str):
        """Update usage count and last used timestamp"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE voice_clones 
            SET usage_count = usage_count + 1, last_used = ?
            WHERE voice_id = ?
        """, (datetime.datetime.now().isoformat(), voice_id))
        
        conn.commit()
        conn.close()
    
    def set_favorite(self, voice_id: str, is_favorite: bool = True):
        """Set or unset voice clone as favorite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE voice_clones 
            SET is_favorite = ?
            WHERE voice_id = ?
        """, (1 if is_favorite else 0, voice_id))
        
        conn.commit()
        conn.close()
    
    def update_rating(self, voice_id: str, rating: int):
        """Update quality rating (1-5 stars)"""
        if not 0 <= rating <= 5:
            return False
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE voice_clones 
            SET quality_rating = ?
            WHERE voice_id = ?
        """, (rating, voice_id))
        
        conn.commit()
        conn.close()
        return True
    
    def delete_voice_clone(self, voice_id: str) -> bool:
        """Delete a voice clone from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM voice_clones WHERE voice_id = ?", (voice_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        return deleted
    
    def get_categories(self) -> List[Dict]:
        """Get all available categories"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name, color, description FROM categories ORDER BY name")
        rows = cursor.fetchall()
        
        categories = []
        for row in rows:
            categories.append({
                'name': row[0],
                'color': row[1],
                'description': row[2]
            })
        
        conn.close()
        return categories
    
    def add_category(self, name: str, color: str = "#3498db", description: str = "") -> bool:
        """Add a new category"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO categories (name, color, description, created_date)
                VALUES (?, ?, ?, ?)
            """, (name, color, description, datetime.datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            return True
            
        except sqlite3.IntegrityError:
            return False
        except Exception as e:
            print(f"Error adding category: {e}")
            return False
    
    def get_voice_clone_stats(self) -> Dict:
        """Get statistics about voice clones"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total count
        cursor.execute("SELECT COUNT(*) FROM voice_clones")
        total_count = cursor.fetchone()[0]
        
        # Count by category
        cursor.execute("""
            SELECT category, COUNT(*) 
            FROM voice_clones 
            GROUP BY category 
            ORDER BY COUNT(*) DESC
        """)
        category_counts = dict(cursor.fetchall())
        
        # Favorites count
        cursor.execute("SELECT COUNT(*) FROM voice_clones WHERE is_favorite = 1")
        favorites_count = cursor.fetchone()[0]
        
        # Most used
        cursor.execute("""
            SELECT name, usage_count 
            FROM voice_clones 
            WHERE usage_count > 0 
            ORDER BY usage_count DESC 
            LIMIT 5
        """)
        most_used = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_count': total_count,
            'category_counts': category_counts,
            'favorites_count': favorites_count,
            'most_used': most_used
        }
    
    def export_voice_clones(self) -> Dict:
        """Export all voice clone data"""
        voice_clones = self.get_voice_clones()
        categories = self.get_categories()
        
        return {
            'voice_clones': voice_clones,
            'categories': categories,
            'exported_at': datetime.datetime.now().isoformat()
        }
    
    def import_voice_clones(self, data: Dict) -> bool:
        """Import voice clone data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Import categories first
            if 'categories' in data:
                for category in data['categories']:
                    cursor.execute("""
                        INSERT OR IGNORE INTO categories (name, color, description, created_date)
                        VALUES (?, ?, ?, ?)
                    """, (category['name'], category.get('color', '#3498db'), 
                         category.get('description', ''), datetime.datetime.now().isoformat()))
            
            # Import voice clones
            if 'voice_clones' in data:
                for voice in data['voice_clones']:
                    cursor.execute("""
                        INSERT OR REPLACE INTO voice_clones 
                        (voice_id, name, category, description, tags, created_date, 
                         last_used, usage_count, quality_rating, is_favorite, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (voice['voice_id'], voice['name'], voice['category'], 
                         voice.get('description', ''), json.dumps(voice.get('tags', [])),
                         voice['created_date'], voice.get('last_used'), 
                         voice.get('usage_count', 0), voice.get('quality_rating', 0),
                         1 if voice.get('is_favorite', False) else 0,
                         json.dumps(voice.get('metadata', {}))))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error importing voice clones: {e}")
            return False