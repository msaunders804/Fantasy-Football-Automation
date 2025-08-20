# pfr_collector.py
"""
Pro Football Reference Data Collector
Scrapes advanced stats and historical data from Pro Football Reference
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import time
import re
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from database_setup import FantasyDatabase
import urllib.parse

class PFRCollector:
    """Collects and stores data from Pro Football Reference"""
    
    def __init__(self, db_path="fantasy_football.db"):
        self.base_url = "https://www.pro-football-reference.com"
        self.db = FantasyDatabase(db_path)
        self.session = requests.Session()
        self.setup_logging()
        
        # Rate limiting - Be very respectful to PFR
        self.rate_limit_delay = 3  # 3 seconds between requests
        self.last_request_time = 0
        
        # Headers to appear more like a browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    def setup_logging(self):
        """Configure logging for PFR collector"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _rate_limit(self):
        """Implement respectful rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            self.logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, url: str) -> Optional[BeautifulSoup]:
        """Make web request with error handling and parsing"""
        self._rate_limit()
        
        try:
            self.logger.debug(f"Requesting: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Check if we're being blocked or rate limited
            if "blocked" in soup.get_text().lower() or response.status_code == 429:
                self.logger.warning("Possible rate limiting detected, extending delay")
                time.sleep(10)
                return None
            
            return soup
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed for {url}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Parsing failed for {url}: {e}")
            return None
    
    def _parse_player_name(self, name_cell) -> Tuple[str, str]:
        """Parse player name and extract player ID if available"""
        if not name_cell:
            return "", ""
        
        # Try to find player link for ID
        player_link = name_cell.find('a')
        player_id = ""
        
        if player_link and 'href' in player_link.attrs:
            href = player_link['href']
            # Extract player ID from URL like /players/A/AlleJo02.htm
            match = re.search(r'/players/\w/(\w+)\.htm', href)
            if match:
                player_id = match.group(1)
        
        # Get player name text
        name_text = name_cell.get_text(strip=True)
        # Remove any asterisks or other symbols
        clean_name = re.sub(r'[*+]', '', name_text).strip()
        
        return clean_name, player_id
    
    def _safe_float(self, value) -> float:
        """Safely convert value to float"""
        if not value or value in ['', '--', 'N/A']:
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _safe_int(self, value) -> int:
        """Safely convert value to int"""
        if not value or value in ['', '--', 'N/A']:
            return 0
        try:
            return int(float(value))  # Convert to float first to handle decimals
        except (ValueError, TypeError):
            return 0
    
    def collect_passing_stats(self, season: int) -> bool:
        """Collect passing statistics for a season"""
        start_time = datetime.now()
        self.logger.info(f"Collecting {season} passing stats from PFR...")
        
        url = f"{self.base_url}/years/{season}/passing.htm"
        soup = self._make_request(url)
        
        if not soup:
            self.logger.error(f"Failed to fetch passing stats for {season}")
            return False
        
        if not self.db.connect():
            return False
        
        try:
            # Find the main passing stats table
            table = soup.find('table', {'id': 'passing'})
            if not table:
                self.logger.error("Could not find passing table")
                return False
            
            tbody = table.find('tbody')
            if not tbody:
                self.logger.error("Could not find table body")
                return False
            
            cursor = self.db.connection.cursor()
            records_inserted = 0
            
            for row in tbody.find_all('tr'):
                # Skip header rows and divider rows
                if row.get('class') and ('thead' in row.get('class') or 'divider' in row.get('class')):
                    continue
                
                cells = row.find_all(['td', 'th'])
                if len(cells) < 15:  # Need minimum columns for meaningful data
                    continue
                
                try:
                    # Parse basic info
                    rank_cell = cells[0] if len(cells) > 0 else None
                    name_cell = cells[1] if len(cells) > 1 else None
                    team_cell = cells[2] if len(cells) > 2 else None
                    
                    if not name_cell:
                        continue
                    
                    player_name, pfr_player_id = self._parse_player_name(name_cell)
                    team = team_cell.get_text(strip=True) if team_cell else ""
                    
                    # Skip if no meaningful name
                    if not player_name or len(player_name) < 2:
                        continue
                    
                    # Parse passing stats (typical PFR passing table columns)
                    age = self._safe_int(cells[3].get_text(strip=True)) if len(cells) > 3 else 0
                    games = self._safe_int(cells[4].get_text(strip=True)) if len(cells) > 4 else 0
                    games_started = self._safe_int(cells[5].get_text(strip=True)) if len(cells) > 5 else 0
                    qb_record = cells[6].get_text(strip=True) if len(cells) > 6 else ""
                    completions = self._safe_int(cells[7].get_text(strip=True)) if len(cells) > 7 else 0
                    attempts = self._safe_int(cells[8].get_text(strip=True)) if len(cells) > 8 else 0
                    completion_pct = self._safe_float(cells[9].get_text(strip=True)) if len(cells) > 9 else 0
                    pass_yards = self._safe_int(cells[10].get_text(strip=True)) if len(cells) > 10 else 0
                    pass_tds = self._safe_int(cells[11].get_text(strip=True)) if len(cells) > 11 else 0
                    td_pct = self._safe_float(cells[12].get_text(strip=True)) if len(cells) > 12 else 0
                    interceptions = self._safe_int(cells[13].get_text(strip=True)) if len(cells) > 13 else 0
                    int_pct = self._safe_float(cells[14].get_text(strip=True)) if len(cells) > 14 else 0
                    
                    # Additional stats if available
                    longest = self._safe_int(cells[15].get_text(strip=True)) if len(cells) > 15 else 0
                    yards_per_attempt = self._safe_float(cells[16].get_text(strip=True)) if len(cells) > 16 else 0
                    adj_yards_per_attempt = self._safe_float(cells[17].get_text(strip=True)) if len(cells) > 17 else 0
                    yards_per_completion = self._safe_float(cells[18].get_text(strip=True)) if len(cells) > 18 else 0
                    passer_rating = self._safe_float(cells[19].get_text(strip=True)) if len(cells) > 19 else 0
                    qbr = self._safe_float(cells[20].get_text(strip=True)) if len(cells) > 20 else 0
                    sacks = self._safe_int(cells[21].get_text(strip=True)) if len(cells) > 21 else 0
                    sack_yards = self._safe_int(cells[22].get_text(strip=True)) if len(cells) > 22 else 0
                    
                    # Try to match player to our existing database
                    cursor.execute("""
                        SELECT player_id FROM players 
                        WHERE (full_name LIKE ? OR full_name LIKE ?) 
                        AND position = 'QB'
                        LIMIT 1
                    """, (f"%{player_name}%", f"{player_name}%"))
                    
                    player_match = cursor.fetchone()
                    if not player_match:
                        self.logger.debug(f"No player match found for QB: {player_name}")
                        continue
                    
                    player_id = player_match[0]
                    
                    # Insert advanced stats
                    cursor.execute("""
                        INSERT OR REPLACE INTO advanced_stats (
                            player_id, season_year, week,
                            completion_pct_above_exp, yards_after_catch_per_completion,
                            passer_rating, qbr, air_yards_per_attempt,
                            data_source, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        player_id, season, 0,  # Week 0 indicates season totals
                        completion_pct, yards_per_completion, 
                        passer_rating, qbr, yards_per_attempt,
                        'pfr', datetime.now()
                    ))
                    
                    records_inserted += 1
                    
                    if records_inserted % 10 == 0:
                        self.logger.info(f"Processed {records_inserted} QB records...")
                        self.db.connection.commit()
                
                except Exception as e:
                    self.logger.error(f"Error processing passing row: {e}")
                    continue
            
            self.db.connection.commit()
            
            end_time = datetime.now()
            self.logger.info(f"Successfully collected {records_inserted} passing records for {season}")
            
            # Log the collection
            self.db.log_data_collection(
                data_source='pfr',
                collection_type='passing_stats',
                season_year=season,
                records_collected=records_inserted,
                success=True,
                started_at=start_time,
                completed_at=end_time
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in passing stats collection: {e}")
            
            # Log failed collection
            self.db.log_data_collection(
                data_source='pfr',
                collection_type='passing_stats',
                season_year=season,
                records_collected=0,
                success=False,
                error_message=str(e),
                started_at=start_time,
                completed_at=datetime.now()
            )
            return False
        finally:
            self.db.disconnect()
    
    def collect_rushing_stats(self, season: int) -> bool:
        """Collect rushing statistics for a season"""
        start_time = datetime.now()
        self.logger.info(f"Collecting {season} rushing stats from PFR...")
        
        url = f"{self.base_url}/years/{season}/rushing.htm"
        soup = self._make_request(url)
        
        if not soup:
            self.logger.error(f"Failed to fetch rushing stats for {season}")
            return False
        
        if not self.db.connect():
            return False
        
        try:
            # Find the main rushing stats table
            table = soup.find('table', {'id': 'rushing'})
            if not table:
                self.logger.error("Could not find rushing table")
                return False
            
            tbody = table.find('tbody')
            if not tbody:
                self.logger.error("Could not find table body")
                return False
            
            cursor = self.db.connection.cursor()
            records_inserted = 0
            
            for row in tbody.find_all('tr'):
                # Skip header and divider rows
                if row.get('class') and ('thead' in row.get('class') or 'divider' in row.get('class')):
                    continue
                
                cells = row.find_all(['td', 'th'])
                if len(cells) < 10:  # Need minimum columns
                    continue
                
                try:
                    # Parse basic info
                    name_cell = cells[1] if len(cells) > 1 else None
                    team_cell = cells[2] if len(cells) > 2 else None
                    
                    if not name_cell:
                        continue
                    
                    player_name, pfr_player_id = self._parse_player_name(name_cell)
                    team = team_cell.get_text(strip=True) if team_cell else ""
                    
                    if not player_name or len(player_name) < 2:
                        continue
                    
                    # Parse rushing stats
                    age = self._safe_int(cells[3].get_text(strip=True)) if len(cells) > 3 else 0
                    games = self._safe_int(cells[4].get_text(strip=True)) if len(cells) > 4 else 0
                    games_started = self._safe_int(cells[5].get_text(strip=True)) if len(cells) > 5 else 0
                    rush_attempts = self._safe_int(cells[6].get_text(strip=True)) if len(cells) > 6 else 0
                    rush_yards = self._safe_int(cells[7].get_text(strip=True)) if len(cells) > 7 else 0
                    rush_tds = self._safe_int(cells[8].get_text(strip=True)) if len(cells) > 8 else 0
                    longest_rush = self._safe_int(cells[9].get_text(strip=True)) if len(cells) > 9 else 0
                    yards_per_attempt = self._safe_float(cells[10].get_text(strip=True)) if len(cells) > 10 else 0
                    yards_per_game = self._safe_float(cells[11].get_text(strip=True)) if len(cells) > 11 else 0
                    fumbles = self._safe_int(cells[12].get_text(strip=True)) if len(cells) > 12 else 0
                    
                    # Try to match player to our existing database
                    cursor.execute("""
                        SELECT player_id FROM players 
                        WHERE (full_name LIKE ? OR full_name LIKE ?) 
                        AND position IN ('RB', 'QB', 'WR', 'FB')
                        LIMIT 1
                    """, (f"%{player_name}%", f"{player_name}%"))
                    
                    player_match = cursor.fetchone()
                    if not player_match:
                        self.logger.debug(f"No player match found for rusher: {player_name}")
                        continue
                    
                    player_id = player_match[0]
                    
                    # Calculate advanced rushing metrics (simplified)
                    if rush_attempts > 0:
                        rush_success_rate = min(1.0, yards_per_attempt / 4.0)  # Simplified calculation
                        yards_before_contact = yards_per_attempt * 0.6  # Estimate
                        yards_after_contact = yards_per_attempt * 0.4   # Estimate
                    else:
                        rush_success_rate = 0.0
                        yards_before_contact = 0.0
                        yards_after_contact = 0.0
                    
                    # Insert advanced stats
                    cursor.execute("""
                        INSERT OR REPLACE INTO advanced_stats (
                            player_id, season_year, week,
                            yards_before_contact_per_att, yards_after_contact_per_att,
                            rush_success_rate, broken_tackles,
                            data_source, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        player_id, season, 0,  # Week 0 indicates season totals
                        yards_before_contact, yards_after_contact,
                        rush_success_rate, 0,  # Broken tackles not available in basic table
                        'pfr', datetime.now()
                    ))
                    
                    records_inserted += 1
                    
                    if records_inserted % 10 == 0:
                        self.logger.info(f"Processed {records_inserted} rushing records...")
                        self.db.connection.commit()
                
                except Exception as e:
                    self.logger.error(f"Error processing rushing row: {e}")
                    continue
            
            self.db.connection.commit()
            
            end_time = datetime.now()
            self.logger.info(f"Successfully collected {records_inserted} rushing records for {season}")
            
            # Log the collection
            self.db.log_data_collection(
                data_source='pfr',
                collection_type='rushing_stats',
                season_year=season,
                records_collected=records_inserted,
                success=True,
                started_at=start_time,
                completed_at=end_time
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in rushing stats collection: {e}")
            
            # Log failed collection
            self.db.log_data_collection(
                data_source='pfr',
                collection_type='rushing_stats',
                season_year=season,
                records_collected=0,
                success=False,
                error_message=str(e),
                started_at=start_time,
                completed_at=datetime.now()
            )
            return False
        finally:
            self.db.disconnect()
    
    def collect_receiving_stats(self, season: int) -> bool:
        """Collect receiving statistics for a season"""
        start_time = datetime.now()
        self.logger.info(f"Collecting {season} receiving stats from PFR...")
        
        url = f"{self.base_url}/years/{season}/receiving.htm"
        soup = self._make_request(url)
        
        if not soup:
            self.logger.error(f"Failed to fetch receiving stats for {season}")
            return False
        
        if not self.db.connect():
            return False
        
        try:
            # Find the main receiving stats table
            table = soup.find('table', {'id': 'receiving'})
            if not table:
                self.logger.error("Could not find receiving table")
                return False
            
            tbody = table.find('tbody')
            if not tbody:
                self.logger.error("Could not find table body")
                return False
            
            cursor = self.db.connection.cursor()
            records_inserted = 0
            
            for row in tbody.find_all('tr'):
                # Skip header and divider rows
                if row.get('class') and ('thead' in row.get('class') or 'divider' in row.get('class')):
                    continue
                
                cells = row.find_all(['td', 'th'])
                if len(cells) < 10:  # Need minimum columns
                    continue
                
                try:
                    # Parse basic info
                    name_cell = cells[1] if len(cells) > 1 else None
                    team_cell = cells[2] if len(cells) > 2 else None
                    
                    if not name_cell:
                        continue
                    
                    player_name, pfr_player_id = self._parse_player_name(name_cell)
                    team = team_cell.get_text(strip=True) if team_cell else ""
                    
                    if not player_name or len(player_name) < 2:
                        continue
                    
                    # Parse receiving stats
                    age = self._safe_int(cells[3].get_text(strip=True)) if len(cells) > 3 else 0
                    games = self._safe_int(cells[4].get_text(strip=True)) if len(cells) > 4 else 0
                    games_started = self._safe_int(cells[5].get_text(strip=True)) if len(cells) > 5 else 0
                    targets = self._safe_int(cells[6].get_text(strip=True)) if len(cells) > 6 else 0
                    receptions = self._safe_int(cells[7].get_text(strip=True)) if len(cells) > 7 else 0
                    rec_yards = self._safe_int(cells[8].get_text(strip=True)) if len(cells) > 8 else 0
                    yards_per_reception = self._safe_float(cells[9].get_text(strip=True)) if len(cells) > 9 else 0
                    rec_tds = self._safe_int(cells[10].get_text(strip=True)) if len(cells) > 10 else 0
                    longest_rec = self._safe_int(cells[11].get_text(strip=True)) if len(cells) > 11 else 0
                    targets_per_game = self._safe_float(cells[12].get_text(strip=True)) if len(cells) > 12 else 0
                    rec_per_game = self._safe_float(cells[13].get_text(strip=True)) if len(cells) > 13 else 0
                    rec_yards_per_game = self._safe_float(cells[14].get_text(strip=True)) if len(cells) > 14 else 0
                    catch_percentage = self._safe_float(cells[15].get_text(strip=True)) if len(cells) > 15 else 0
                    
                    # Try to match player to our existing database
                    cursor.execute("""
                        SELECT player_id FROM players 
                        WHERE (full_name LIKE ? OR full_name LIKE ?) 
                        AND position IN ('WR', 'TE', 'RB', 'QB')
                        LIMIT 1
                    """, (f"%{player_name}%", f"{player_name}%"))
                    
                    player_match = cursor.fetchone()
                    if not player_match:
                        self.logger.debug(f"No player match found for receiver: {player_name}")
                        continue
                    
                    player_id = player_match[0]
                    
                    # Calculate advanced receiving metrics (simplified)
                    if targets > 0:
                        catch_rate = receptions / targets
                        drop_rate = 1 - catch_rate
                        target_share = targets / games if games > 0 else 0
                    else:
                        catch_rate = 0.0
                        drop_rate = 0.0
                        target_share = 0.0
                    
                    if receptions > 0:
                        yards_after_catch = yards_per_reception * 0.4  # Estimate
                        separation = 2.5  # Default estimate
                    else:
                        yards_after_catch = 0.0
                        separation = 0.0
                    
                    # Insert advanced stats
                    cursor.execute("""
                        INSERT OR REPLACE INTO advanced_stats (
                            player_id, season_year, week,
                            catch_rate, drop_rate, yards_after_catch,
                            separation, contested_catch_rate,
                            data_source, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        player_id, season, 0,  # Week 0 indicates season totals
                        catch_rate, drop_rate, yards_after_catch,
                        separation, 0.0,  # Contested catch rate not available in basic table
                        'pfr', datetime.now()
                    ))
                    
                    records_inserted += 1
                    
                    if records_inserted % 10 == 0:
                        self.logger.info(f"Processed {records_inserted} receiving records...")
                        self.db.connection.commit()
                
                except Exception as e:
                    self.logger.error(f"Error processing receiving row: {e}")
                    continue
            
            self.db.connection.commit()
            
            end_time = datetime.now()
            self.logger.info(f"Successfully collected {records_inserted} receiving records for {season}")
            
            # Log the collection
            self.db.log_data_collection(
                data_source='pfr',
                collection_type='receiving_stats',
                season_year=season,
                records_collected=records_inserted,
                success=True,
                started_at=start_time,
                completed_at=end_time
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in receiving stats collection: {e}")
            
            # Log failed collection
            self.db.log_data_collection(
                data_source='pfr',
                collection_type='receiving_stats',
                season_year=season,
                records_collected=0,
                success=False,
                error_message=str(e),
                started_at=start_time,
                completed_at=datetime.now()
            )
            return False
        finally:
            self.db.disconnect()
    
    def collect_team_offense_stats(self, season: int) -> bool:
        """Collect team offensive statistics"""
        start_time = datetime.now()
        self.logger.info(f"Collecting {season} team offense stats from PFR...")
        
        url = f"{self.base_url}/years/{season}/opp.htm"
        soup = self._make_request(url)
        
        if not soup:
            self.logger.error(f"Failed to fetch team stats for {season}")
            return False
        
        if not self.db.connect():
            return False
        
        try:
            # Find the team stats table
            table = soup.find('table', {'id': 'team_stats'})
            if not table:
                self.logger.error("Could not find team stats table")
                return False
            
            tbody = table.find('tbody')
            if not tbody:
                self.logger.error("Could not find table body")
                return False
            
            cursor = self.db.connection.cursor()
            records_inserted = 0
            
            for row in tbody.find_all('tr'):
                cells = row.find_all(['td', 'th'])
                if len(cells) < 10:
                    continue
                
                try:
                    # Parse team info
                    team_cell = cells[1] if len(cells) > 1 else None
                    if not team_cell:
                        continue
                    
                    team_name = team_cell.get_text(strip=True)
                    
                    # Map team name to abbreviation (simplified)
                    team_mapping = {
                        'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL',
                        'Baltimore Ravens': 'BAL', 'Buffalo Bills': 'BUF',
                        'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
                        'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE',
                        'Dallas Cowboys': 'DAL', 'Denver Broncos': 'DEN',
                        'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
                        'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND',
                        'Jacksonville Jaguars': 'JAX', 'Kansas City Chiefs': 'KC',
                        'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
                        'Los Angeles Rams': 'LAR', 'Miami Dolphins': 'MIA',
                        'Minnesota Vikings': 'MIN', 'New England Patriots': 'NE',
                        'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
                        'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI',
                        'Pittsburgh Steelers': 'PIT', 'San Francisco 49ers': 'SF',
                        'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
                        'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'
                    }
                    
                    team_abbr = team_mapping.get(team_name, team_name[:3].upper())
                    
                    # Parse offensive stats
                    games = self._safe_int(cells[2].get_text(strip=True)) if len(cells) > 2 else 0
                    points = self._safe_int(cells[3].get_text(strip=True)) if len(cells) > 3 else 0
                    total_yards = self._safe_int(cells[4].get_text(strip=True)) if len(cells) > 4 else 0
                    pass_yards = self._safe_int(cells[5].get_text(strip=True)) if len(cells) > 5 else 0
                    rush_yards = self._safe_int(cells[6].get_text(strip=True)) if len(cells) > 6 else 0
                    turnovers = self._safe_int(cells[7].get_text(strip=True)) if len(cells) > 7 else 0
                    
                    # Calculate per-game averages
                    points_per_game = points / games if games > 0 else 0
                    yards_per_game = total_yards / games if games > 0 else 0
                    pass_yd_per_game = pass_yards / games if games > 0 else 0
                    rush_yd_per_game = rush_yards / games if games > 0 else 0
                    turnovers_per_game = turnovers / games if games > 0 else 0
                    
                    # Insert team stats
                    cursor.execute("""
                        INSERT OR REPLACE INTO team_stats (
                            team_abbr, season_year, week,
                            points_per_game, yards_per_game, pass_yd_per_game,
                            rush_yd_per_game, turnovers_per_game,
                            data_source, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        team_abbr, season, 0,  # Week 0 indicates season totals
                        points_per_game, yards_per_game, pass_yd_per_game,
                        rush_yd_per_game, turnovers_per_game,
                        'pfr', datetime.now()
                    ))
                    
                    records_inserted += 1
                
                except Exception as e:
                    self.logger.error(f"Error processing team row: {e}")
                    continue
            
            self.db.connection.commit()
            
            end_time = datetime.now()
            self.logger.info(f"Successfully collected {records_inserted} team records for {season}")
            
            # Log the collection
            self.db.log_data_collection(
                data_source='pfr',
                collection_type='team_stats',
                season_year=season,
                records_collected=records_inserted,
                success=True,
                started_at=start_time,
                completed_at=end_time
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in team stats collection: {e}")
            return False
        finally:
            self.db.disconnect()
    
    def collect_season_data(self, season: int) -> bool:
        """Collect all available data for a season"""
        self.logger.info(f"Starting comprehensive data collection for {season} season...")
        
        success_count = 0
        total_collections = 4
        
        # Collect passing stats
        if self.collect_passing_stats(season):
            success_count += 1
            self.logger.info(f"✅ Passing stats collected for {season}")
        else:
            self.logger.error(f"❌ Passing stats failed for {season}")
        
        # Collect rushing stats
        if self.collect_rushing_stats(season):
            success_count += 1
            self.logger.info(f"✅ Rushing stats collected for {season}")
        else:
            self.logger.error(f"❌ Rushing stats failed for {season}")
        
        # Collect receiving stats
        if self.collect_receiving_stats(season):
            success_count += 1
            self.logger.info(f"✅ Receiving stats collected for {season}")
        else:
            self.logger.error(f"❌ Receiving stats failed for {season}")
        
        # Collect team stats
        if self.collect_team_offense_stats(season):
            success_count += 1
            self.logger.info(f"✅ Team stats collected for {season}")
        else:
            self.logger.error(f"❌ Team stats failed for {season}")
        
        self.logger.info(f"Season {season} collection complete: {success_count}/{total_collections} successful")
        return success_count >= (total_collections * 0.75)  # 75% success rate required
    
    def collect_historical_data(self, seasons: List[int] = None) -> bool:
        """Collect historical data for multiple seasons"""
        if seasons is None:
            seasons = [2021, 2022, 2023]
        
        self.logger.info(f"Starting historical PFR data collection for seasons: {seasons}")
        
        success_count = 0
        for season in seasons:
            self.logger.info(f"Collecting PFR data for {season} season...")
            
            if self.collect_season_data(season):
                success_count += 1
                self.logger.info(f"✅ {season} season completed")
            else:
                self.logger.error(f"❌ {season} season had issues")
            
            # Longer pause between seasons to be respectful
            if season != seasons[-1]:  # Don't sleep after last season
                self.logger.info("Pausing between seasons...")
                time.sleep(30)
        
        self.logger.info(f"Historical PFR collection complete: {success_count}/{len(seasons)} seasons successful")
        return success_count > 0

def main():
    """Main execution for PFR data collection"""
    print("🏈 PRO FOOTBALL REFERENCE DATA COLLECTOR")
    print("=" * 50)
    print("⚠️  IMPORTANT: This scraper is respectful but slow")
    print("⚠️  Please use responsibly and consider PFR's resources")
    print("=" * 50)
    
    collector = PFRCollector()
    
    while True:
        print(f"\n" + "=" * 50)
        print("COLLECTION OPTIONS:")
        print("1. Collect passing stats for season")
        print("2. Collect rushing stats for season")
        print("3. Collect receiving stats for season")
        print("4. Collect team stats for season")
        print("5. Collect all stats for season")
        print("6. Collect historical data (2021-2023)")
        print("7. Exit")
        print("=" * 50)
        
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == '1':
            season = int(input("Enter season year (e.g., 2023): "))
            print(f"\n📊 Collecting {season} passing stats...")
            
            if collector.collect_passing_stats(season):
                print("✅ Passing stats collected successfully")
            else:
                print("❌ Passing stats collection failed")
        
        elif choice == '2':
            season = int(input("Enter season year (e.g., 2023): "))
            print(f"\n🏃 Collecting {season} rushing stats...")
            
            if collector.collect_rushing_stats(season):
                print("✅ Rushing stats collected successfully")
            else:
                print("❌ Rushing stats collection failed")
        
        elif choice == '3':
            season = int(input("Enter season year (e.g., 2023): "))
            print(f"\n🎯 Collecting {season} receiving stats...")
            
            if collector.collect_receiving_stats(season):
                print("✅ Receiving stats collected successfully")
            else:
                print("❌ Receiving stats collection failed")
        
        elif choice == '4':
            season = int(input("Enter season year (e.g., 2023): "))
            print(f"\n🏈 Collecting {season} team stats...")
            
            if collector.collect_team_offense_stats(season):
                print("✅ Team stats collected successfully")
            else:
                print("❌ Team stats collection failed")
        
        elif choice == '5':
            season = int(input("Enter season year (e.g., 2023): "))
            print(f"\n📅 Collecting all {season} stats...")
            print("This may take 10-15 minutes due to rate limiting...")
            
            if collector.collect_season_data(season):
                print("✅ Season data collection completed")
            else:
                print("❌ Season data collection had issues")
        
        elif choice == '6':
            print("\n📚 Collecting historical data (2021-2023)...")
            print("⚠️  This will take 30-45 minutes due to rate limiting")
            print("⚠️  Please be patient and respectful to PFR servers")
            
            confirm = input("Continue? (y/N): ").strip().lower()
            if confirm == 'y':
                if collector.collect_historical_data():
                    print("✅ Historical data collection completed")
                else:
                    print("❌ Historical data collection had issues")
            else:
                print("Historical collection cancelled")
        
        elif choice == '7':
            print("👋 Exiting PFR collector")
            break
        
        else:
            print("❌ Invalid option, please try again")

if __name__ == "__main__":
    main()