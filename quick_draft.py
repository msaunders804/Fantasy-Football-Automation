# quick_draft_tool.py
"""
Quick Draft Assistant
Simplified interface for immediate draft help using your collected data
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List
from database_setup import FantasyDatabase

class QuickDraftAssistant:
    """Simplified draft assistant using statistical analysis"""
    
    def __init__(self, db_path="fantasy_football.db"):
        self.db = FantasyDatabase(db_path)
        self.league_settings = {
            'teams': 12,
            'scoring': 'ppr',
            'roster_spots': {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1}
        }
        self.drafted_players = set()
        self.my_roster = {'QB': [], 'RB': [], 'WR': [], 'TE': []}
    
    def get_player_projections(self) -> List[Dict]:
        """Get 2025 statistical projections for all active players"""
        if not self.db.connect():
            return []
        
        try:
            # Query for recent performance with focus on 2024 season for 2025 projections
            query = """
            WITH player_recent AS (
                SELECT 
                    p.player_id,
                    p.full_name,
                    p.position,
                    p.team,
                    p.age,
                    p.status,
                    -- 2024 stats (most recent)
                    COUNT(CASE WHEN ps.season_year = 2024 THEN ps.week END) as games_played_2024,
                    SUM(CASE WHEN ps.season_year = 2024 THEN ps.fantasy_points_ppr END) as total_points_2024,
                    AVG(CASE WHEN ps.season_year = 2024 THEN ps.fantasy_points_ppr END) as avg_points_2024,
                    SUM(CASE WHEN ps.season_year = 2024 THEN ps.rec_tgt END) as total_targets_2024,
                    SUM(CASE WHEN ps.season_year = 2024 THEN ps.rush_att END) as total_rushes_2024,
                    STDEV(CASE WHEN ps.season_year = 2024 THEN ps.fantasy_points_ppr END) as points_std_2024,
                    -- 2023 stats for context
                    COUNT(CASE WHEN ps.season_year = 2023 THEN ps.week END) as games_played_2023,
                    SUM(CASE WHEN ps.season_year = 2023 THEN ps.fantasy_points_ppr END) as total_points_2023,
                    AVG(CASE WHEN ps.season_year = 2023 THEN ps.fantasy_points_ppr END) as avg_points_2023,
                    -- 2022 for trend analysis
                    AVG(CASE WHEN ps.season_year = 2022 THEN ps.fantasy_points_ppr END) as avg_points_2022
                FROM players p
                LEFT JOIN player_stats ps ON p.player_id = ps.player_id 
                    AND ps.season_year >= 2022
                WHERE p.position IN ('QB', 'RB', 'WR', 'TE')
                    AND (p.status = 'Active' OR p.status IS NULL)
                    AND p.team IS NOT NULL
                    AND p.team != ''
                    AND p.team != 'FA'
                    -- Filter out clearly retired players
                    AND p.full_name NOT LIKE '%Tom Brady%'
                    AND p.full_name NOT LIKE '%Ben Roethlisberger%'
                    AND p.full_name NOT LIKE '%Josh McCown%'
                    AND p.full_name NOT LIKE '%Adrian Peterson%'
                    AND p.full_name NOT LIKE '%Frank Gore%'
                    AND p.full_name NOT LIKE '%Antonio Brown%'
                    AND (p.age IS NULL OR p.age < 38)
                GROUP BY p.player_id, p.full_name, p.position, p.team, p.age, p.status
            )
            SELECT * FROM player_recent
            WHERE (games_played_2024 > 0 OR games_played_2023 > 4 OR (age < 25))
            ORDER BY COALESCE(total_points_2024, total_points_2023, 0) DESC
            """
            
            df = pd.read_sql_query(query, self.db.connection)
            
            projections = []
            
            for _, row in df.iterrows():
                # Get most recent performance data
                points_2024 = row['total_points_2024'] or 0
                points_2023 = row['total_points_2023'] or 0
                games_2024 = row['games_played_2024'] or 0
                games_2023 = row['games_played_2023'] or 0
                
                # Calculate per-game averages
                ppg_2024 = (points_2024 / games_2024) if games_2024 > 0 else 0
                ppg_2023 = (points_2023 / games_2023) if games_2023 > 0 else 0
                ppg_2022 = row['avg_points_2022'] or 0
                
                # Project 2025 performance using weighted approach
                if ppg_2024 > 5:  # Good 2024 data
                    # 70% 2024, 20% 2023, 10% 2022
                    projected_ppg = (ppg_2024 * 0.7) + (ppg_2023 * 0.2) + (ppg_2022 * 0.1)
                elif ppg_2023 > 5:  # Good 2023 data
                    # 60% 2023, 30% 2022, 10% position baseline
                    position_baseline = self._get_position_baseline(row['position'])
                    projected_ppg = (ppg_2023 * 0.6) + (ppg_2022 * 0.3) + (position_baseline * 0.1)
                elif ppg_2022 > 5:  # Only 2022 data
                    projected_ppg = ppg_2022 * 0.85  # Discount older data
                else:
                    # New/rookie player - use position baseline
                    projected_ppg = self._get_position_baseline(row['position'])
                
                # Age and experience adjustments
                age = row['age'] or 25
                age_factor = self._calculate_age_factor(age, row['position'])
                projected_ppg *= age_factor
                
                # Project to 17-game season
                projected_season = projected_ppg * 17
                
                # Apply position-specific floors and ceilings
                projected_season = self._apply_position_bounds(projected_season, row['position'])
                
                # Skip players with very low projections unless they're very young
                if projected_season < 50 and age > 24:
                    continue
                
                # Consistency score
                std_dev = row['points_std_2024'] or (projected_ppg * 0.4)
                consistency = 1 / (1 + (std_dev / projected_ppg)) if projected_ppg > 0 else 0.5
                
                projections.append({
                    'player_id': row['player_id'],
                    'name': row['full_name'],
                    'position': row['position'],
                    'team': row['team'],
                    'age': age,
                    'projected_points': round(projected_season, 1),
                    'projected_ppg': round(projected_ppg, 1),
                    'consistency': round(consistency, 3),
                    'games_2024': games_2024,
                    'ppg_2024': round(ppg_2024, 1),
                    'ppg_2023': round(ppg_2023, 1),
                    'total_targets': row['total_targets_2024'] or 0,
                    'total_rushes': row['total_rushes_2024'] or 0
                })
            
            return projections
            
        except Exception as e:
            print(f"Error getting projections: {e}")
            return []
        finally:
            self.db.disconnect()
    
    def _get_position_baseline(self, position: str) -> float:
        """Get baseline PPG for position (for rookies/unknowns)"""
        baselines = {
            'QB': 18.0,  # ~300 season points
            'RB': 10.0,  # ~170 season points  
            'WR': 8.0,   # ~135 season points
            'TE': 6.0    # ~100 season points
        }
        return baselines.get(position, 5.0)
    
    def _calculate_age_factor(self, age: int, position: str) -> float:
        """Calculate age-based performance factor"""
        if position == 'QB':
            if age <= 25:
                return 1.05  # Young QB upside
            elif age <= 32:
                return 1.0   # Prime years
            elif age <= 36:
                return 0.95  # Slight decline
            else:
                return 0.85  # Significant decline
        
        elif position == 'RB':
            if age <= 24:
                return 1.1   # Young RB peak
            elif age <= 27:
                return 1.0   # Prime years
            elif age <= 30:
                return 0.9   # Decline starts
            else:
                return 0.75  # Sharp decline
        
        elif position in ['WR', 'TE']:
            if age <= 26:
                return 1.05  # Young receiver upside
            elif age <= 31:
                return 1.0   # Prime years
            elif age <= 34:
                return 0.95  # Slight decline
            else:
                return 0.85  # Notable decline
        
        return 1.0
    
    def _apply_position_bounds(self, projection: float, position: str) -> float:
        """Apply reasonable bounds by position"""
        bounds = {
            'QB': (120, 420),  # Range for QBs
            'RB': (40, 350),   # Range for RBs
            'WR': (30, 320),   # Range for WRs
            'TE': (25, 200)    # Range for TEs
        }
        
        min_proj, max_proj = bounds.get(position, (20, 400))
        return max(min_proj, min(max_proj, projection))
                elif ppg_2023 > 0:
                    projected_ppg = ppg_2023
                elif ppg_2022 > 0:
                    projected_ppg = ppg_2022 * 0.8  # Discount if no 2023 data
                else:
                    # Rookie/new player estimation
                    if row['position'] == 'QB':
                        projected_ppg = 15
                    elif row['position'] == 'RB':
                        projected_ppg = 8
                    elif row['position'] == 'WR':
                        projected_ppg = 7
                    elif row['position'] == 'TE':
                        projected_ppg = 5
                    else:
                        projected_ppg = 3
                
                # Age adjustment
                age = row['age'] or 25
                if age <= 24:
                    age_factor = 1.05  # Young player bonus
                elif age <= 28:
                    age_factor = 1.0   # Prime years
                elif age <= 31:
                    age_factor = 0.95  # Slight decline
                else:
                    age_factor = 0.9   # Veteran decline
                
                projected_ppg *= age_factor
                
                # Project to 17-game season
                projected_season = projected_ppg * 17
                
                # Consistency score (lower std deviation = more consistent)
                consistency = 1 / (1 + (row['points_std'] or 10))
                
                projections.append({
                    'player_id': row['player_id'],
                    'name': row['full_name'],
                    'position': row['position'],
                    'team': row['team'],
                    'age': age,
                    'projected_points': round(projected_season, 1),
                    'projected_ppg': round(projected_ppg, 1),
                    'consistency': round(consistency, 3),
                    'games_2023': games_2023,
                    'ppg_2023': round(ppg_2023, 1),
                    'ppg_2022': round(ppg_2022, 1),
                    'total_targets': row['total_targets'] or 0,
                    'total_rushes': row['total_rushes'] or 0
                })
            
            return projections
            
        except Exception as e:
            print(f"Error getting projections: {e}")
            return []
        finally:
            self.db.disconnect()
    
    def calculate_vbd(self, projections: List[Dict]) -> List[Dict]:
        """Calculate Value-Based Drafting scores"""
        # Determine replacement levels
        replacement_levels = {}
        teams = self.league_settings['teams']
        
        for position in ['QB', 'RB', 'WR', 'TE']:
            pos_players = [p for p in projections if p['position'] == position]
            pos_players.sort(key=lambda x: x['projected_points'], reverse=True)
            
            if position == 'QB':
                replacement_idx = teams  # 12th QB
            elif position == 'RB':
                replacement_idx = teams * 3  # ~36th RB (RB2 * 12 teams + handcuffs)
            elif position == 'WR':
                replacement_idx = teams * 3  # ~36th WR
            elif position == 'TE':
                replacement_idx = teams * 2  # ~24th TE
            
            if len(pos_players) > replacement_idx:
                replacement_levels[position] = pos_players[replacement_idx]['projected_points']
            else:
                replacement_levels[position] = pos_players[-1]['projected_points'] if pos_players else 0
        
        # Calculate VBD scores
        for projection in projections:
            position = projection['position']
            projected_points = projection['projected_points']
            replacement_level = replacement_levels.get(position, 0)
            
            vbd_score = max(0, projected_points - replacement_level)
            projection['vbd_score'] = round(vbd_score, 1)
            projection['replacement_level'] = round(replacement_level, 1)
        
        return projections
    
    def get_positional_needs(self) -> Dict[str, int]:
        """Calculate current roster needs"""
        needs = {}
        for position, required in self.league_settings['roster_spots'].items():
            if position == 'FLEX':
                continue
            current = len(self.my_roster.get(position, []))
            needs[position] = max(0, required - current)
        
        # Calculate FLEX need
        flex_eligible = len(self.my_roster['RB']) + len(self.my_roster['WR']) + len(self.my_roster['TE'])
        total_flex_spots = self.league_settings['roster_spots']['RB'] + self.league_settings['roster_spots']['WR'] + self.league_settings['roster_spots']['TE'] + self.league_settings['roster_spots']['FLEX']
        
        if flex_eligible < total_flex_spots:
            needs['FLEX'] = total_flex_spots - flex_eligible
        
        return needs
    
    def get_draft_recommendations(self, projections: List[Dict], top_n: int = 10) -> List[Dict]:
        """Get top draft recommendations"""
        # Filter out drafted players
        available = [p for p in projections if p['player_id'] not in self.drafted_players]
        
        # Sort by VBD score
        available.sort(key=lambda x: x['vbd_score'], reverse=True)
        
        # Get positional needs
        needs = self.get_positional_needs()
        
        # Score recommendations
        recommendations = []
        for player in available[:50]:  # Consider top 50 by VBD
            position = player['position']
            
            # Base score is VBD
            score = player['vbd_score']
            
            # Boost for positional need
            if needs.get(position, 0) > 0:
                score *= 1.3  # 30% boost for needed positions
            elif position in ['RB', 'WR', 'TE'] and needs.get('FLEX', 0) > 0:
                score *= 1.1  # 10% boost for flex-eligible
            
            # Boost for consistency
            score *= (1 + player['consistency'])
            
            # Determine recommendation type
            if needs.get(position, 0) > 0:
                rec_type = "NEED FILL"
            elif player['vbd_score'] > 50:
                rec_type = "ELITE VALUE"
            elif player['vbd_score'] > 25:
                rec_type = "GOOD VALUE"
            else:
                rec_type = "BPA"
            
            recommendations.append({
                **player,
                'draft_score': round(score, 1),
                'recommendation_type': rec_type
            })
        
        # Sort by draft score and return top N
        recommendations.sort(key=lambda x: x['draft_score'], reverse=True)
        return recommendations[:top_n]
    
    def draft_player(self, player_name: str, projections: List[Dict]) -> bool:
        """Draft a player to your roster"""
        # Find player
        matches = [p for p in projections if player_name.lower() in p['name'].lower()]
        
        if not matches:
            return False
        
        player = matches[0]
        position = player['position']
        
        # Add to roster
        if position not in self.my_roster:
            self.my_roster[position] = []
        
        self.my_roster[position].append(player)
        self.drafted_players.add(player['player_id'])
        
        return True
    
    def show_roster(self):
        """Display current roster"""
        print(f"\n📋 YOUR ROSTER:")
        print("-" * 50)
        
        total_projected = 0
        
        for position in ['QB', 'RB', 'WR', 'TE']:
            players = self.my_roster.get(position, [])
            if players:
                print(f"\n{position}:")
                for i, player in enumerate(players, 1):
                    print(f"  {i}. {player['name']} ({player['team']}) - {player['projected_points']} pts")
                    total_projected += player['projected_points']
            else:
                print(f"\n{position}: (empty)")
        
        print(f"\nTotal Projected Points: {total_projected:.1f}")

def main():
    """Quick draft interface"""
    print("⚡ QUICK FANTASY DRAFT ASSISTANT")
    print("=" * 40)
    print("Using your collected data for instant draft help!")
    
    assistant = QuickDraftAssistant()
    
    # Setup
    try:
        teams = int(input("\nNumber of teams (default 12): ") or "12")
        scoring = input("Scoring (ppr/half_ppr/standard, default ppr): ").strip().lower() or "ppr"
        
        assistant.league_settings['teams'] = teams
        assistant.league_settings['scoring'] = scoring
        
        print(f"\n✅ Setup: {teams} teams, {scoring.upper()} scoring")
    except ValueError:
        print("Using default settings: 12 teams, PPR scoring")
    
    # Get projections
    print("\n📊 Loading player projections...")
    projections = assistant.get_player_projections()
    
    if not projections:
        print("❌ No player data available. Please collect data first.")
        return
    
    # Calculate VBD
    projections = assistant.calculate_vbd(projections)
    
    print(f"✅ Loaded {len(projections)} players")
    
    # Main draft loop
    while True:
        print(f"\n" + "=" * 50)
        print("DRAFT COMMANDS:")
        print("'rec' - Get recommendations")
        print("'rank [position]' - Show positional rankings (e.g., 'rank RB')")
        print("'draft [player name]' - Draft a player")
        print("'roster' - Show your roster")
        print("'search [name]' - Search for specific player")
        print("'quit' - Exit")
        print("=" * 50)
        
        command = input("\nCommand: ").strip().lower()
        
        if command == 'quit':
            break
        
        elif command == 'rec':
            recommendations = assistant.get_draft_recommendations(projections)
            
            print(f"\n⭐ TOP DRAFT RECOMMENDATIONS:")
            print("-" * 80)
            print(f"{'#':<2} {'Player':<22} {'Pos':<3} {'Team':<4} {'Proj':<6} {'VBD':<6} {'Type':<12}")
            print("-" * 80)
            
            for i, rec in enumerate(recommendations, 1):
                print(f"{i:<2} {rec['name'][:21]:<22} {rec['position']:<3} "
                      f"{rec['team'] or 'FA':<4} {rec['projected_points']:<6.1f} "
                      f"{rec['vbd_score']:<6.1f} {rec['recommendation_type']:<12}")
        
        elif command.startswith('rank '):
            position = command.split(' ')[1].upper()
            
            if position in ['QB', 'RB', 'WR', 'TE']:
                pos_players = [p for p in projections if p['position'] == position and p['player_id'] not in assistant.drafted_players]
                pos_players.sort(key=lambda x: x['projected_points'], reverse=True)
                
                print(f"\n🏆 {position} RANKINGS:")
                print("-" * 60)
                print(f"{'#':<3} {'Player':<25} {'Team':<4} {'Proj':<6} {'2023':<6}")
                print("-" * 60)
                
                for i, player in enumerate(pos_players[:15], 1):
                    print(f"{i:<3} {player['name'][:24]:<25} {player['team'] or 'FA':<4} "
                          f"{player['projected_points']:<6.1f} {player['ppg_2023']:<6.1f}")
            else:
                print("❌ Invalid position. Use QB, RB, WR, or TE")
        
        elif command.startswith('draft '):
            player_name = command[6:].strip()
            
            if assistant.draft_player(player_name, projections):
                player = next(p for p in projections if player_name.lower() in p['name'].lower())
                print(f"✅ Drafted {player['name']} ({player['position']}) - {player['projected_points']} pts")
                assistant.show_roster()
            else:
                print(f"❌ Player '{player_name}' not found")
        
        elif command == 'roster':
            assistant.show_roster()
        
        elif command.startswith('search '):
            search_term = command[7:].strip().lower()
            matches = [p for p in projections if search_term in p['name'].lower()]
            
            if matches:
                print(f"\n🔍 Search results for '{search_term}':")
                print("-" * 60)
                for player in matches[:10]:
                    status = "❌ DRAFTED" if player['player_id'] in assistant.drafted_players else "✅ Available"
                    print(f"  {player['name']} ({player['position']}, {player['team']}) - "
                          f"{player['projected_points']} pts - {status}")
            else:
                print(f"❌ No players found matching '{search_term}'")
        
        else:
            print("❌ Unknown command")

if __name__ == "__main__":
    main()