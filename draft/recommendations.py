import numpy as np
from typing import List, Dict, Any, Optional
from core.database import DatabaseManager

class RecommendationEngine:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def get_recommendations(self, draft_id: int, team_position: int = None, limit: int = 10) -> List[Dict]:
        """Get ML-powered draft recommendations"""
        try:
            # Get draft context
            draft_context = self._get_draft_context(draft_id, team_position)
            if not draft_context:
                return []
            
            # Get available players
            available_players = self._get_available_players(draft_id)
            
            if not available_players:
                return []
            
            # Calculate recommendations for each player
            recommendations = []
            
            for player in available_players:
                rec_score = self._calculate_recommendation_score(player, draft_context)
                
                recommendations.append({
                    'player_id': player['player_id'],
                    'name': player['name'],
                    'position': player['position'],
                    'team': player.get('team'),
                    'vbd_score': player.get('vbd_score', 0) or 0,
                    'projected_points': player.get('final_points', 0) or 0,
                    'tier': player.get('tier', 6) or 6,
                    'recommendation_score': rec_score['total_score'],
                    'value': rec_score['value_score'],
                    'positional_need': rec_score['need_score'],
                    'scarcity_factor': rec_score['scarcity_score'],
                    'reason': rec_score['reason']
                })
            
            # Sort by recommendation score
            recommendations.sort(key=lambda x: x['recommendation_score'], reverse=True)
            
            return recommendations[:limit]
            
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return []
    
    def _get_draft_context(self, draft_id: int, team_position: int = None) -> Dict:
        """Get current draft context"""
        # Get draft info
        draft_query = """
        SELECT d.*, l.num_teams, l.scoring_system, l.roster_settings
        FROM drafts d
        JOIN leagues l ON d.league_id = l.league_id
        WHERE d.draft_id = ?
        """
        draft_result = self.db.execute_query(draft_query, (draft_id,))
        
        if not draft_result:
            return None
        
        draft_info = draft_result[0]
        current_pick = draft_info['current_pick']
        round_num = ((current_pick - 1) // draft_info['num_teams']) + 1
        
        # Determine team position if not provided
        if team_position is None:
            team_position = self._get_team_position(current_pick, draft_info['num_teams'])
        
        # Get current team roster
        roster = self._get_team_roster(draft_id, team_position)
        
        # Get positional needs
        needs = self._calculate_positional_needs(roster, draft_info['roster_settings'])
        
        # Get draft trends
        trends = self._analyze_draft_trends(draft_id, round_num)
        
        return {
            'draft_id': draft_id,
            'current_pick': current_pick,
            'round_number': round_num,
            'team_position': team_position,
            'num_teams': draft_info['num_teams'],
            'total_rounds': draft_info['total_rounds'],
            'roster': roster,
            'positional_needs': needs,
            'draft_trends': trends,
            'picks_until_next_turn': self._picks_until_next_turn(current_pick, team_position, draft_info['num_teams'])
        }
    
    def _get_available_players(self, draft_id: int) -> List[Dict]:
        """Get all available players with projections"""
        query = """
        SELECT p.*, COALESCE(ap.final_points, 0) as final_points, 
               COALESCE(ap.vbd_score, 0) as vbd_score, 
               COALESCE(ap.tier, 6) as tier, 
               COALESCE(ap.confidence_score, 0.5) as confidence_score
        FROM players p
        LEFT JOIN aggregated_projections ap ON p.player_id = ap.player_id AND ap.week = 0
        WHERE p.player_id NOT IN (
            SELECT DISTINCT player_id FROM draft_picks 
            WHERE draft_id = ? AND player_id IS NOT NULL
        )
        ORDER BY COALESCE(ap.vbd_score, 0) DESC
        """
        return self.db.execute_query(query, (draft_id,))
    
    def _get_team_roster(self, draft_id: int, team_position: int) -> List[Dict]:
        """Get current team roster"""
        query = """
        SELECT p.*, tr.position_slot
        FROM team_rosters tr
        JOIN players p ON tr.player_id = p.player_id
        WHERE tr.draft_id = ? AND tr.team_position = ?
        """
        return self.db.execute_query(query, (draft_id, team_position))
    
    def _calculate_positional_needs(self, roster: List[Dict], roster_settings: str) -> Dict:
        """Calculate positional needs based on current roster"""
        import json
        
        try:
            required_positions = json.loads(roster_settings)
        except:
            # Default roster if parsing fails
            required_positions = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1, 'K': 1, 'DEF': 1}
        
        current_positions = {}
        
        # Count current positions
        for player in roster:
            pos = player['position']
            current_positions[pos] = current_positions.get(pos, 0) + 1
        
        # Calculate needs (higher score = more needed)
        needs = {}
        for pos, required in required_positions.items():
            if pos == 'BENCH':
                continue
            
            current = current_positions.get(pos, 0)
            
            if pos == 'FLEX':
                # FLEX can be filled by RB/WR/TE
                flex_eligible = sum(current_positions.get(p, 0) for p in ['RB', 'WR', 'TE'])
                base_positions_needed = sum(required_positions.get(p, 0) for p in ['RB', 'WR', 'TE'])
                needs[pos] = max(0, required - max(0, flex_eligible - base_positions_needed))
            else:
                needs[pos] = max(0, required - current)
        
        # Normalize to 0-1 scale
        max_need = max(needs.values()) if needs.values() else 1
        if max_need > 0:
            needs = {pos: need / max_need for pos, need in needs.items()}
        
        return needs
    
    def _analyze_draft_trends(self, draft_id: int, current_round: int) -> Dict:
        """Analyze how draft is trending by position"""
        query = """
        SELECT p.position, COUNT(*) as count, AVG(dp.pick_number) as avg_pick
        FROM draft_picks dp
        JOIN players p ON dp.player_id = p.player_id
        WHERE dp.draft_id = ? AND dp.round_number <= ?
        GROUP BY p.position
        """
        
        trends = self.db.execute_query(query, (draft_id, current_round))
        
        # Calculate run indicators
        recent_picks_query = """
        SELECT p.position
        FROM draft_picks dp
        JOIN players p ON dp.player_id = p.player_id
        WHERE dp.draft_id = ? AND dp.pick_number >= ?
        ORDER BY dp.pick_number DESC
        LIMIT 6
        """
        
        start_pick = max(1, (current_round - 1) * 12)
        recent_picks = self.db.execute_query(recent_picks_query, (draft_id, start_pick))
        
        # Detect position runs
        position_runs = {}
        if recent_picks:
            last_positions = [pick['position'] for pick in recent_picks]
            for pos in ['QB', 'RB', 'WR', 'TE']:
                position_runs[pos] = last_positions.count(pos)
        
        return {
            'position_counts': {trend['position']: trend['count'] for trend in trends},
            'position_runs': position_runs
        }
    
    def _picks_until_next_turn(self, current_pick: int, team_position: int, num_teams: int) -> int:
        """Calculate picks until this team's next turn"""
        current_round = ((current_pick - 1) // num_teams) + 1
        current_team_pick = self._get_team_position(current_pick, num_teams)
        
        if current_team_pick == team_position:
            return 0
        
        # Calculate next pick for this team
        if current_round % 2 == 1:  # Odd round
            if team_position > current_team_pick:
                return team_position - current_team_pick
            else:
                return (num_teams - current_team_pick) + (num_teams - team_position + 1)
        else:  # Even round
            reverse_team_pos = num_teams - team_position + 1
            reverse_current = num_teams - current_team_pick + 1
            
            if reverse_team_pos > reverse_current:
                return reverse_team_pos - reverse_current
            else:
                return reverse_current + team_position
    
    def _get_team_position(self, pick_number: int, num_teams: int) -> int:
        """Calculate team position for pick number"""
        round_num = ((pick_number - 1) // num_teams) + 1
        pos_in_round = ((pick_number - 1) % num_teams) + 1
        
        if round_num % 2 == 1:
            return pos_in_round
        else:
            return num_teams - pos_in_round + 1
    
    def _calculate_recommendation_score(self, player: Dict, draft_context: Dict) -> Dict:
        """Calculate overall recommendation score for a player"""
        # Base value score (VBD)
        value_score = player.get('vbd_score', 0) or 0
        
        # Positional need score
        position = player['position']
        need_score = draft_context['positional_needs'].get(position, 0)
        
        # Handle FLEX eligibility
        if position in ['RB', 'WR', 'TE']:
            flex_need = draft_context['positional_needs'].get('FLEX', 0)
            need_score = max(need_score, flex_need * 0.8)
        
        # Scarcity/timing score
        scarcity_score = self._calculate_scarcity_score(player, draft_context)
        
        # Tier considerations
        tier_score = max(0, 6 - (player.get('tier', 6) or 6)) / 5
        
        # Combine scores with weights
        total_score = (
            value_score * 0.4 +          # 40% VBD
            need_score * 100 * 0.3 +     # 30% positional need
            scarcity_score * 0.2 +       # 20% scarcity
            tier_score * 20 * 0.1        # 10% tier
        )
        
        # Generate reason
        reason = self._generate_recommendation_reason(player, draft_context, {
            'value_score': value_score,
            'need_score': need_score,
            'scarcity_score': scarcity_score,
            'tier_score': tier_score
        })
        
        return {
            'total_score': total_score,
            'value_score': value_score,
            'need_score': need_score,
            'scarcity_score': scarcity_score,
            'tier_score': tier_score,
            'reason': reason
        }
    
    def _calculate_scarcity_score(self, player: Dict, draft_context: Dict) -> float:
        """Calculate scarcity/timing score"""
        position = player['position']
        picks_until_next = draft_context['picks_until_next_turn']
        
        # Check if there's a position run
        position_runs = draft_context['draft_trends']['position_runs']
        if position_runs.get(position, 0) >= 2:
            return 50
        
        # Check tier scarcity
        tier = player.get('tier', 6) or 6
        if tier <= 2 and picks_until_next > 20:
            return 30
        
        # Standard scarcity based on picks until next turn
        if picks_until_next <= 5:
            return 0
        elif picks_until_next <= 15:
            return 10
        else:
            return 25
    
    def _generate_recommendation_reason(self, player: Dict, draft_context: Dict, scores: Dict) -> str:
        """Generate human-readable reason for recommendation"""
        reasons = []
        
        # Value reason
        if scores['value_score'] >= 20:
            reasons.append("Excellent value")
        elif scores['value_score'] >= 10:
            reasons.append("Good value")
        
        # Need reason
        if scores['need_score'] >= 0.8:
            reasons.append(f"Critical {player['position']} need")
        elif scores['need_score'] >= 0.5:
            reasons.append(f"Positional need")
        
        # Scarcity reason
        if scores['scarcity_score'] >= 30:
            reasons.append("Position run happening")
        elif scores['scarcity_score'] >= 20:
            reasons.append("Long wait until next pick")
        
        # Tier reason
        tier = player.get('tier', 6) or 6
        if tier <= 2:
            reasons.append(f"Tier {tier} player")
        
        return "; ".join(reasons) if reasons else "Solid pick"