"""
Enhanced Database migration script for multi-provider support.
Extends the original migrate_db.py to add support for Anthropic, Google, and Tavily API keys.

Run this BEFORE deploying the new multi-provider code.
"""
from sqlalchemy import create_engine, text
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings

def migrate_database():
    """
    Add multi-provider API key columns to users table.
    Also adds provider tracking to threads and messages tables.
    """
    engine = create_engine(settings.database_url)
    
    print("=" * 70)
    print("DATABASE MIGRATION: Multi-Provider Support")
    print("=" * 70)
    print()
    
    changes_made = False
    
    try:
        with engine.connect() as conn:
            # ================================================================
            # STEP 1: Migrate Users Table
            # ================================================================
            print("📋 STEP 1: Migrating Users Table")
            print("-" * 70)
            
            # Check if old column exists and needs renaming
            result = conn.execute(text("""
                SELECT COUNT(*) 
                FROM pragma_table_info('users') 
                WHERE name='encrypted_api_key'
            """))
            has_old_column = result.scalar() > 0
            
            result = conn.execute(text("""
                SELECT COUNT(*) 
                FROM pragma_table_info('users') 
                WHERE name='encrypted_openai_key'
            """))
            has_new_column = result.scalar() > 0
            
            # Rename old column if it exists
            if has_old_column and not has_new_column:
                print("📝 Renaming 'encrypted_api_key' → 'encrypted_openai_key'...")
                try:
                    conn.execute(text("""
                        ALTER TABLE users 
                        RENAME COLUMN encrypted_api_key TO encrypted_openai_key
                    """))
                    conn.commit()
                    print("✅ Column renamed successfully")
                    changes_made = True
                except Exception as e:
                    # SQLite version might not support RENAME COLUMN
                    print(f"⚠️  Could not rename column: {e}")
                    print("   You may need to manually migrate data")
            elif has_new_column:
                print("✅ Column 'encrypted_openai_key' already exists")
            
            # Add new provider API key columns
            new_user_columns = [
                ("encrypted_openai_key", "TEXT", "OpenAI API key storage"),
                ("encrypted_anthropic_key", "TEXT", "Anthropic (Claude) API key storage"),
                ("encrypted_google_key", "TEXT", "Google (Gemini) API key storage"),
                ("encrypted_tavily_key", "TEXT", "Tavily (Web Search) API key storage"),
                ("default_provider", "TEXT DEFAULT 'openai'", "User's preferred AI provider"),
                ("updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP", "Last update timestamp"),
            ]
            
            for column_name, column_type, description in new_user_columns:
                result = conn.execute(text(f"""
                    SELECT COUNT(*) 
                    FROM pragma_table_info('users') 
                    WHERE name='{column_name}'
                """))
                
                if result.scalar() == 0:
                    print(f"📝 Adding '{column_name}' ({description})...")
                    try:
                        conn.execute(text(f"""
                            ALTER TABLE users 
                            ADD COLUMN {column_name} {column_type}
                        """))
                        conn.commit()
                        print(f"✅ Added '{column_name}'")
                        changes_made = True
                    except Exception as e:
                        print(f"❌ Failed to add '{column_name}': {e}")
                else:
                    print(f"✅ Column '{column_name}' already exists")
            
            print()
            
            # ================================================================
            # STEP 2: Migrate Threads Table
            # ================================================================
            print("📋 STEP 2: Migrating Threads Table")
            print("-" * 70)
            
            new_thread_columns = [
                ("active_provider", "TEXT DEFAULT 'openai'", "Current AI provider for thread"),
                ("temperature", "TEXT DEFAULT '1.0'", "Temperature setting"),
                ("system_prompt", "TEXT", "Custom system instructions"),
            ]
            
            for column_name, column_type, description in new_thread_columns:
                result = conn.execute(text(f"""
                    SELECT COUNT(*) 
                    FROM pragma_table_info('threads') 
                    WHERE name='{column_name}'
                """))
                
                if result.scalar() == 0:
                    print(f"📝 Adding '{column_name}' ({description})...")
                    try:
                        conn.execute(text(f"""
                            ALTER TABLE threads 
                            ADD COLUMN {column_name} {column_type}
                        """))
                        conn.commit()
                        print(f"✅ Added '{column_name}'")
                        changes_made = True
                    except Exception as e:
                        print(f"❌ Failed to add '{column_name}': {e}")
                else:
                    print(f"✅ Column '{column_name}' already exists")
            
            print()
            
            # ================================================================
            # STEP 3: Migrate Messages Table
            # ================================================================
            print("📋 STEP 3: Migrating Messages Table")
            print("-" * 70)
            
            new_message_columns = [
                ("provider_used", "TEXT DEFAULT 'openai'", "AI provider that generated this message"),
                ("file_size", "INTEGER", "Size of attached file"),
                ("file_type", "TEXT", "Type of attached file"),
                ("prompt_tokens", "INTEGER", "Tokens used in prompt"),
                ("completion_tokens", "INTEGER", "Tokens used in completion"),
                ("total_tokens", "INTEGER", "Total tokens used"),
            ]
            
            for column_name, column_type, description in new_message_columns:
                result = conn.execute(text(f"""
                    SELECT COUNT(*) 
                    FROM pragma_table_info('messages') 
                    WHERE name='{column_name}'
                """))
                
                if result.scalar() == 0:
                    print(f"📝 Adding '{column_name}' ({description})...")
                    try:
                        conn.execute(text(f"""
                            ALTER TABLE messages 
                            ADD COLUMN {column_name} {column_type}
                        """))
                        conn.commit()
                        print(f"✅ Added '{column_name}'")
                        changes_made = True
                    except Exception as e:
                        print(f"❌ Failed to add '{column_name}': {e}")
                else:
                    print(f"✅ Column '{column_name}' already exists")
            
            print()
            
            # ================================================================
            # STEP 4: Migrate Projects Table (if exists)
            # ================================================================
            print("📋 STEP 4: Migrating Projects Table (if exists)")
            print("-" * 70)
            
            # Check if projects table exists
            result = conn.execute(text("""
                SELECT COUNT(*) 
                FROM sqlite_master 
                WHERE type='table' AND name='projects'
            """))
            
            if result.scalar() > 0:
                new_project_columns = [
                    ("description", "TEXT", "Project description"),
                    ("updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP", "Last update timestamp"),
                ]
                
                for column_name, column_type, description in new_project_columns:
                    result = conn.execute(text(f"""
                        SELECT COUNT(*) 
                        FROM pragma_table_info('projects') 
                        WHERE name='{column_name}'
                    """))
                    
                    if result.scalar() == 0:
                        print(f"📝 Adding '{column_name}' ({description})...")
                        try:
                            conn.execute(text(f"""
                                ALTER TABLE projects 
                                ADD COLUMN {column_name} {column_type}
                            """))
                            conn.commit()
                            print(f"✅ Added '{column_name}'")
                            changes_made = True
                        except Exception as e:
                            print(f"❌ Failed to add '{column_name}': {e}")
                    else:
                        print(f"✅ Column '{column_name}' already exists")
            else:
                print("⚠️  Projects table doesn't exist - skipping")
            
            print()

            # ================================================================
            # STEP 5: Add Location and Profile Fields 
            # ================================================================
            print("📋 STEP 5: Adding Location and Profile Fields")
            print("-" * 70)
            
            new_location_columns = [
                ("location_city", "VARCHAR(255)", "User's city"),
                ("location_state", "VARCHAR(255)", "User's state/province"),
                ("location_country", "VARCHAR(255)", "User's country"),
                ("location_latitude", "VARCHAR(50)", "Latitude coordinate"),
                ("location_longitude", "VARCHAR(50)", "Longitude coordinate"),
                ("location_timezone", "VARCHAR(100)", "User's timezone"),
                ("location_formatted", "TEXT", "Formatted location string"),
                ("location_updated_at", "TIMESTAMP", "When location was last updated"),
                ("name", "VARCHAR(255)", "User's name"),
                ("occupation", "VARCHAR(255)", "User's occupation"),
                ("preferences", "TEXT", "User preferences and interests"),
            ]
            
            for column_name, column_type, description in new_location_columns:
                result = conn.execute(text(f"""
                    SELECT COUNT(*) 
                    FROM pragma_table_info('users') 
                    WHERE name='{column_name}'
                """))
                
                if result.scalar() == 0:
                    print(f"📝 Adding '{column_name}' ({description})...")
                    try:
                        conn.execute(text(f"""
                            ALTER TABLE users 
                            ADD COLUMN {column_name} {column_type}
                        """))
                        conn.commit()
                        print(f"✅ Added '{column_name}'")
                        changes_made = True
                    except Exception as e:
                        print(f"❌ Failed to add '{column_name}': {e}")
                else:
                    print(f"✅ Column '{column_name}' already exists")
            
            print()

            
            # ================================================================
            # SUMMARY
            # ================================================================
            print("=" * 70)
            if changes_made:
                print("✅ MIGRATION COMPLETED SUCCESSFULLY!")
            else:
                print("✅ DATABASE ALREADY UP TO DATE!")
            print("=" * 70)
            print()
            
            if changes_made:
                print("📦 Next Steps:")
                print("1. Install new dependencies:")
                print("   pip install anthropic google-generativeai")
                print()
                print("2. Deploy updated backend code")
                print()
                print("3. Users can now add multiple provider API keys:")
                print("   - OpenAI (GPT-4o, GPT-4, GPT-3.5)")
                print("   - Anthropic (Claude 3.5 Sonnet, Opus, Haiku)")
                print("   - Google (Gemini 2.0 Flash, 1.5 Pro/Flash)")
                print("   - Tavily (Web search - optional)")
                print()
            else:
                print("ℹ️  No changes needed - your database is already configured")
                print("   for multi-provider support!")
                print()
            
            return True
            
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ MIGRATION FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        print()
        print("📝 Troubleshooting:")
        print("1. Make sure the database file exists")
        print("2. Check that you have write permissions")
        print("3. Ensure no other process is using the database")
        print("4. Try running with Python 3.10+")
        print()
        return False


if __name__ == "__main__":
    print()
    print("🚀 Starting database migration...")
    print()
    
    success = migrate_database()
    
    if success:
        print("=" * 70)
        print("🎉 Your MemoryChat is now ready for multi-provider support!")
        print("=" * 70)
        print()
    
    sys.exit(0 if success else 1)