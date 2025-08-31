import requests
import time
from typing import Dict, List, Optional, Any
from config.settings import SLEEPER_BASE_URL

class SleeperAPI:
    """Sleeper fantasy football API client"""
    
    def __init__(self):
        self.base_url = SLEEPER_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Fantasy-Draft-Assistant/1.0'
        })
    
    def get_league_info(self, league_id: str) -> Optional[Dict]:
        """Get league information"""
        try:
            url = f"{self.base_url}/league/{league_id}"
            response = self.session.get(url)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error getting league info: {e}")
            return None
    
    def get_league_users(self, league_id: str) -> List[Dict]:
        """Get league users/members"""
        try:
            url = f"{self.base_url}/league/{league_id}/users"
            response = self.session.get(url)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error getting league users: {e}")
            return []
    
    def get_league_drafts(self, league_id: str) -> List[Dict]:
        """Get all drafts for a league"""
        try:
            url = f"{self.base_url}/league/{league_id}/drafts"
            response = self.session.get(url)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error getting league drafts: {e}")
            return []
    
    def get_draft_info(self, draft_id: str) -> Optional[Dict]:
        """Get draft information"""
        try:
            url = f"{self.base_url}/draft/{draft_id}"
            response = self.session.get(url)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error getting draft info: {e}")
            return None
    
    def get_draft_picks(self, draft_id: str) -> List[Dict]:
        """Get all picks from a draft"""
        try:
            url = f"{self.base_url}/draft/{draft_id}/picks"
            response = self.session.get(url)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error getting draft picks: {e}")
            return []
    
    def get_league_draft_info(self, league_id: str) -> Optional[Dict]:
        """Get current/most recent draft info for league"""
        drafts = self.get_league_drafts(league_id)
        
        if not drafts:
            return None
        
        # Get most recent draft
        current_draft = max(drafts, key=lambda d: d.get('created', 0))
        
        # Get detailed draft info
        draft_info = self.get_draft_info(current_draft['draft_id'])
        
        if draft_info:
            # Add picks
            draft_info['picks'] = self.get_draft_picks(current_draft['draft_id'])
            
            # Add league settings
            league_info = self.get_league_info(league_id)
            if league_info:
                draft_info['scoring_settings'] = league_info.get('scoring_settings', {})
                draft_info['roster_positions'] = league_info.get('roster_positions', [])
        
        return draft_info
    
    def get_player_info(self, player_id: str) -> Optional[Dict]:
        """Get player information (requires player endpoint)"""
        # Note: Sleeper doesn't have a single player endpoint
        # You'd need to get all players and filter
        return self.get_players().get(player_id)
    
    def get_players(self) -> Dict[str, Dict]:
        """Get all NFL players (cached for season)"""
        try:
            url = f"{self.base_url}/players/nfl"
            response = self.session.get(url)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error getting players: {e}")
            return {}
    
    def map_sleeper_to_internal_player(self, sleeper_player_id: str) -> Optional[str]:
        """Map Sleeper player ID to internal player ID"""
        player_info = self.get_player_info(sleeper_player_id)
        
        if not player_info:
            return None
        
        # Create internal player ID format
        name = f"{player_info.get('first_name', '')} {player_info.get('last_name', '')}".strip()
        position = player_info.get('position', '')
        
        if name and position:
            # Generate same format as used in projection import
            clean_name = ''.join(c for c in name.lower() if c.isalnum())
            return f"{clean_name}_{position.lower()}"
       

    def get_league_settings_mapping(self, league_info: Dict) -> Dict:
        """Convert Sleeper league settings to internal format"""
        scoring_settings = league_info.get('scoring_settings', {})
        roster_positions = league_info.get('roster_positions', [])

        # Map scoring system
        scoring_system = 'standard'
        if scoring_settings.get('rec', 0) == 1.0:
            scoring_system = 'ppr'
        elif scoring_settings.get('rec', 0) == 0.5:
            scoring_system = 'half_ppr'

        # Map roster settings
        roster_mapping = {}
        for pos in roster_positions:
            if pos in roster_mapping:
                roster_mapping[pos] += 1
            else:
                roster_mapping[pos] = 1

        return {
            'num_teams': league_info.get('total_rosters', 12),
            'scoring_system': scoring_system,
            'roster_settings': roster_mapping,
            'league_name': league_info.get('name', 'Sleeper League')
        }

    def monitor_draft_updates(self, draft_id: str, last_pick_number: int = 0, 
                        callback=None) -> List[Dict]:
        """Monitor for new draft picks since last check"""
        new_picks = []

        try:
            current_picks = self.get_draft_picks(draft_id)
            
            for pick in current_picks:
                if pick.get('pick_no', 0) > last_pick_number:
                    new_picks.append(pick)
                    
                    if callback:
                        callback(pick)
            
            return new_picks
            
        except Exception as e:
            print(f"Error monitoring draft updates: {e}")
            return []

    def is_draft_active(self, draft_id: str) -> bool:
        """Check if draft is currently active"""
        draft_info = self.get_draft_info(draft_id)

        if not draft_info:
            return False

        status = draft_info.get('status', '')
        return status in ['drafting', 'in_progress']

    def get_current_pick_info(self, draft_id: str) -> Optional[Dict]:
        """Get information about current draft pick"""
        draft_info = self.get_draft_info(draft_id)

        if not draft_info:
            return None

        picks = self.get_draft_picks(draft_id)
        current_pick_number = len([p for p in picks if p.get('player_id')]) + 1

        total_picks = draft_info.get('settings', {}).get('rounds', 16) * len(draft_info.get('draft_order', []))

        if current_pick_number > total_picks:
            return None  # Draft is complete

        draft_order = draft_info.get('draft_order', [])
        if not draft_order:
            return None

        # Calculate whose turn it is
        round_number = ((current_pick_number - 1) // len(draft_order)) + 1
        pick_in_round = ((current_pick_number - 1) % len(draft_order)) + 1

        # Handle snake draft
        if round_number % 2 == 0:  # Even rounds go in reverse
            team_index = len(draft_order) - pick_in_round
        else:  # Odd rounds go forward
            team_index = pick_in_round - 1

        current_team_id = draft_order[team_index]

        return {
            'draft_id': draft_id,
            'pick_number': current_pick_number,
            'round_number': round_number,
            'pick_in_round': pick_in_round,
            'team_id': current_team_id,
            'total_picks': total_picks,
            'is_complete': current_pick_number > total_picks
        }