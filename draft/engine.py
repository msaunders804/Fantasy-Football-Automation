import json
import time
from typing import List, Dict, Any, Optional
from core.database import DatabaseManager
from draft.recommendations import RecommendationEngine
from config.settings import DEFAULT_ROSTER, SCORING_SYSTEMS

class DraftEngine:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.recommendation_engine = RecommendationEngine(db_manager)
    
    def create_draft(self, league_id: str, num_teams: int, scoring_system: str, 
                    user_position: int = None, total_rounds: int = 16) -> int:
        """Create a new draft"""
        try:
            # Create league if it doesn't exist
            self._upsert_league(league_id, num_teams, scoring_system)
            
            # Generate draft order
            draft_order = [f"Team_{i+1}" for i in range(num_teams)]
            if user_position and 1 <= user_position <= num_teams:
                draft_order[user_position-1] = "USER"
            
            # Create draft
            query = """
            INSERT INTO drafts (league_id, current_pick, total_rounds, draft_order)
            VALUES (?, 1, ?, ?)
            """
            self.db.execute_update(query, (league_id, total_rounds, json.dumps(draft_order)))
            
            # Get draft ID
            result = self.db.execute_query(
                "SELECT draft_id FROM drafts WHERE league_id = ? ORDER BY created_at DESC LIMIT 1",
                (league_id,)
            )
            return result[0]['draft_id']
            
        except Exception as e:
            raise Exception(f"Failed to create draft: {e}")
    
    def _upsert_league(self, league_id: str, num_teams: int, scoring_system: str):
        """Create or update league settings"""
        query = """
        INSERT INTO leagues (league_id, num_teams, scoring_system, roster_settings)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(league_id) DO UPDATE SET
            num_teams = excluded.num_teams,
            scoring_system = excluded.scoring_system,
            roster_settings = excluded.roster_settings
        """
        roster_json = json.dumps(DEFAULT_ROSTER)
        self.db.execute_update(query, (league_id, num_teams, scoring_system, roster_json))
    
    def make_pick(self, draft_id: int, player_id: str, pick_number: int = None) -> bool:
        """Make a draft pick"""
        try:
            # Get current draft state
            draft_info = self.get_draft_info(draft_id)
            if not draft_info:
                raise ValueError("Draft not found")
            
            current_pick = pick_number or draft_info['current_pick']
            round_num = ((current_pick - 1) // draft_info['num_teams']) + 1
            team_pos = self._get_team_position(current_pick, draft_info['num_teams'])
            
            # Validate pick
            if self._is_player_drafted(draft_id, player_id):
                raise ValueError(f"Player {player_id} has already been drafted")
            
            # Insert pick
            pick_query = """
            INSERT INTO draft_picks (draft_id, pick_number, round_number, team_position, player_id)
            VALUES (?, ?, ?, ?, ?)
            """
            self.db.execute_update(pick_query, (draft_id, current_pick, round_num, team_pos, player_id))
            
            # Update team roster
            player_info = self._get_player_info(player_id)
            position_slot = self._determine_position_slot(draft_id, team_pos, player_info['position'])
            
            roster_query = """
            INSERT INTO team_rosters (draft_id, team_position, player_id, position_slot)
            VALUES (?, ?, ?, ?)
            """
            self.db.execute_update(roster_query, (draft_id, team_pos, player_id, position_slot))
            
            # Update current pick (only if this is the current pick)
            if not pick_number or pick_number == draft_info['current_pick']:
                total_picks = draft_info['total_rounds'] * draft_info['num_teams']
                new_pick = min(current_pick + 1, total_picks + 1)
                
                self.db.execute_update(
                    "UPDATE drafts SET current_pick = ? WHERE draft_id = ?",
                    (new_pick, draft_id)
                )
                
                # Mark draft as complete if finished
                if new_pick > total_picks:
                    self.db.execute_update(
                        "UPDATE drafts SET status = 'completed' WHERE draft_id = ?",
                        (draft_id,)
                    )
            
            return True
            
        except Exception as e:
            raise Exception(f"Failed to make pick: {e}")
    
    def _is_player_drafted(self, draft_id: int, player_id: str) -> bool:
        """Check if player has already been drafted"""
        query = """
        SELECT COUNT(*) as count FROM draft_picks 
        WHERE draft_id = ? AND player_id = ?
        """
        result = self.db.execute_query(query, (draft_id, player_id))
        return result[0]['count'] > 0 if result else False
    
    def _get_player_info(self, player_id: str) -> Dict:
        """Get player information"""
        query = "SELECT * FROM players WHERE player_id = ?"
        result = self.db.execute_query(query, (player_id,))
        return result[0] if result else None
    
    def _determine_position_slot(self, draft_id: int, team_position: int, position: str) -> str:
        """Determine which roster slot this player should fill"""
        # Get current roster for team
        roster = self.get_team_roster(draft_id, team_position)
        position_counts = {}
        
        for player in roster:
            pos = player['position']
            position_counts[pos] = position_counts.get(pos, 0) + 1
        
        # Simple slot assignment logic
        if position == 'QB':
            return f'QB{position_counts.get("QB", 0) + 1}'
        elif position == 'RB':
            return f'RB{position_counts.get("RB", 0) + 1}'
        elif position == 'WR':
            return f'WR{position_counts.get("WR", 0) + 1}'
        elif position == 'TE':
            return f'TE{position_counts.get("TE", 0) + 1}'
        elif position in ['K', 'DEF']:
            return position
        else:
            return 'BENCH'
    
    def _get_team_position(self, pick_number: int, num_teams: int) -> int:
        """Calculate team position for a given pick number (snake draft)"""
        round_num = ((pick_number - 1) // num_teams) + 1
        pos_in_round = ((pick_number - 1) % num_teams) + 1
        
        if round_num % 2 == 1:  # Odd rounds go 1->N
            return pos_in_round
        else:  # Even rounds go N->1 (snake)
            return num_teams - pos_in_round + 1
    
    def get_draft_info(self, draft_id: int) -> Dict:
        """Get current draft information"""
        query = """
        SELECT d.*, l.num_teams, l.scoring_system, l.roster_settings
        FROM drafts d
        JOIN leagues l ON d.league_id = l.league_id
        WHERE d.draft_id = ?
        """
        result = self.db.execute_query(query, (draft_id,))
        
        if not result:
            return None
            
        draft_info = result[0]
        draft_info['draft_order'] = json.loads(draft_info['draft_order'])
        draft_info['roster_settings'] = json.loads(draft_info['roster_settings'])
        
        return draft_info
    
    def get_available_players(self, draft_id: int, position: str = None, limit: int = 50) -> List[Dict]:
        """Get available players for draft"""
        query = """
        SELECT p.*, ap.final_points, ap.vbd_score, ap.tier, ap.confidence_score
        FROM players p
        LEFT JOIN aggregated_projections ap ON p.player_id = ap.player_id AND ap.week = 0
        WHERE p.player_id NOT IN (
            SELECT DISTINCT player_id FROM draft_picks WHERE draft_id = ? AND player_id IS NOT NULL
        )
        """
        params = [draft_id]
        
        if position:
            query += " AND p.position = ?"
            params.append(position)
            
        # Order by VBD if available, otherwise by name
        query += " ORDER BY COALESCE(ap.vbd_score, 0) DESC, p.name"
        
        if limit:
            query += f" LIMIT {limit}"
            
        return self.db.execute_query(query, tuple(params))
    
    def get_team_roster(self, draft_id: int, team_position: int) -> List[Dict]:
        """Get current roster for a team"""
        query = """
        SELECT p.*, tr.position_slot, dp.pick_number, dp.round_number,
               ap.final_points, ap.vbd_score
        FROM team_rosters tr
        JOIN players p ON tr.player_id = p.player_id
        JOIN draft_picks dp ON tr.draft_id = dp.draft_id AND tr.player_id = dp.player_id
        LEFT JOIN aggregated_projections ap ON p.player_id = ap.player_id AND ap.week = 0
        WHERE tr.draft_id = ? AND tr.team_position = ?
        ORDER BY dp.pick_number
        """
        return self.db.execute_query(query, (draft_id, team_position))
    
    def get_recommendations(self, draft_id: int, team_position: int = None) -> List[Dict]:
        """Get ML-powered draft recommendations"""
        return self.recommendation_engine.get_recommendations(draft_id, team_position)
    
    def get_draft_history(self, draft_id: int, limit: int = 20) -> List[Dict]:
        """Get recent draft picks"""
        query = """
        SELECT dp.pick_number, dp.round_number, dp.team_position, 
               p.name, p.position, p.team, d.draft_order,
               ap.final_points, ap.vbd_score
        FROM draft_picks dp
        JOIN players p ON dp.player_id = p.player_id
        JOIN drafts d ON dp.draft_id = d.draft_id
        LEFT JOIN aggregated_projections ap ON p.player_id = ap.player_id AND ap.week = 0
        WHERE dp.draft_id = ?
        ORDER BY dp.pick_number DESC
        LIMIT ?
        """
        
        picks = self.db.execute_query(query, (draft_id, limit))
        
        # Add team names from draft order
        for pick in picks:
            if pick['draft_order']:
                draft_order = json.loads(pick['draft_order'])
                if pick['team_position'] <= len(draft_order):
                    pick['team_name'] = draft_order[pick['team_position'] - 1]
                else:
                    pick['team_name'] = f"Team_{pick['team_position']}"
            else:
                pick['team_name'] = f"Team_{pick['team_position']}"
        
        return picks
    
    def run_draft_loop(self, draft_id: int):
        """Interactive draft loop for manual drafts"""
        import click
        
        draft_info = self.get_draft_info(draft_id)
        if not draft_info:
            raise ValueError("Draft not found")
        
        total_picks = draft_info['total_rounds'] * draft_info['num_teams']
        
        while draft_info['current_pick'] <= total_picks:
            # Refresh draft info
            draft_info = self.get_draft_info(draft_id)
            current_pick = draft_info['current_pick']
            
            if current_pick > total_picks:
                break
                
            round_num = ((current_pick - 1) // draft_info['num_teams']) + 1
            team_pos = self._get_team_position(current_pick, draft_info['num_teams'])
            draft_order = draft_info['draft_order']
            current_team = draft_order[team_pos - 1]
            pick_in_round = ((current_pick - 1) % draft_info['num_teams']) + 1
            
            click.echo(f"\n{'='*60}")
            click.echo(f"🏈 Round {round_num}, Pick {current_pick} ({pick_in_round}/{draft_info['num_teams']})")
            click.echo(f"👥 Current Team: {current_team}")
            click.echo(f"{'='*60}")
            
            # Show team's current roster
            roster = self.get_team_roster(draft_id, team_pos)
            if roster:
                click.echo(f"\n📋 {current_team} Current Roster:")
                for player in roster:
                    points_str = f" ({player['final_points']:.1f})" if player.get('final_points') else ""
                    click.echo(f"  R{player['round_number']}P{player['pick_number']}: "
                              f"{player['position_slot']} - {player['name']} ({player['position']}){points_str}")
            
            # Show recent picks
            recent_picks = self.get_draft_history(draft_id, 5)
            if recent_picks:
                click.echo(f"\n📊 Recent Picks:")
                for pick in recent_picks[:3]:
                    click.echo(f"  P{pick['pick_number']}: {pick['name']} ({pick['position']}) → {pick['team_name']}")
            
            # Show recommendations
            recommendations = self.get_recommendations(draft_id, team_pos)
            if recommendations:
                click.echo(f"\n💡 Top Recommendations for {current_team}:")
                click.echo(f"{'#':<2} {'Name':<20} {'Pos':<3} {'Value':<6} {'Reason':<25}")
                click.echo("-" * 60)
                for i, rec in enumerate(recommendations[:5], 1):
                    click.echo(f"{i:<2} {rec['name']:<20} {rec['position']:<3} "
                              f"{rec['vbd_score']:<6.1f} {rec['reason'][:24]:<25}")
            
            # Get user input
            if current_team == "USER":
                click.echo(f"\n🎯 YOUR PICK!")
                while True:
                    player_input = click.prompt("Enter player name (or command)", type=str).strip()
                    
                    if player_input.lower() in ['help', 'h']:
                        self._show_help()
                        continue
                    elif player_input.lower() in ['quit', 'q', 'exit']:
                        click.echo("Draft session ended")
                        return
                    elif player_input.lower().startswith('available') or player_input.lower().startswith('av'):
                        parts = player_input.split()
                        pos = parts[1].upper() if len(parts) > 1 else None
                        self._show_available_players(draft_id, pos)
                        continue
                    elif player_input.lower().startswith('roster') or player_input.lower().startswith('r'):
                        self._show_team_roster(draft_id, team_pos, current_team)
                        continue
                    elif player_input.lower().startswith('recommend') or player_input.lower().startswith('rec'):
                        self._show_detailed_recommendations(draft_id, team_pos)
                        continue
                    
                    # Try to find player
                    player = self._find_player_by_name(draft_id, player_input)
                    if player:
                        confirm_msg = f"Draft {player['name']} ({player['position']}"
                        if player.get('final_points'):
                            confirm_msg += f", {player['final_points']:.1f} pts"
                        confirm_msg += ")?"
                        
                        if click.confirm(confirm_msg):
                            self.make_pick(draft_id, player['player_id'])
                            click.echo(f"✅ Drafted: {player['name']} ({player['position']})")
                            break
                        else:
                            continue
                    else:
                        click.echo("❌ Player not found or already drafted. Try again or type 'help'")
            else:
                # Simulate CPU pick
                click.echo(f"⏳ {current_team} is making their pick...")
                time.sleep(1.5)  # Brief pause for realism
                
                if recommendations:
                    # CPU picks from top recommendations with some randomness
                    import random
                    top_picks = recommendations[:3]  # Top 3 choices
                    weights = [3, 2, 1]  # Weight towards better picks
                    cpu_pick = random.choices(top_picks, weights=weights)[0]
                    
                    self.make_pick(draft_id, cpu_pick['player_id'])
                    click.echo(f"🤖 {current_team} drafted: {cpu_pick['name']} "
                              f"({cpu_pick['position']}) - VBD: {cpu_pick['vbd_score']:.1f}")
                else:
                    # Fallback if no recommendations available
                    available = self.get_available_players(draft_id, limit=1)
                    if available:
                        self.make_pick(draft_id, available[0]['player_id'])
                        click.echo(f"🤖 {current_team} drafted: {available[0]['name']} ({available[0]['position']})")
        
        click.echo(f"\n🎉 DRAFT COMPLETE! 🎉")
        self._show_final_results(draft_id)
    
    def _show_help(self):
        """Show help commands during draft"""
        import click
        click.echo(f"\n💡 Draft Commands:")
        click.echo(f"  help, h              - Show this help")
        click.echo(f"  available [POS]      - Show available players (optionally by position)")
        click.echo(f"  roster, r            - Show your current roster")
        click.echo(f"  recommend, rec       - Show detailed recommendations")
        click.echo(f"  quit, q              - Exit draft")
        click.echo(f"  [player name]        - Draft a player")
        click.echo(f"\nPositions: QB, RB, WR, TE, K, DEF")
    
    def _show_available_players(self, draft_id: int, position: str = None):
        """Show available players during draft"""
        import click
        
        available = self.get_available_players(draft_id, position, 20)
        
        pos_str = f" {position}" if position else ""
        click.echo(f"\n📋 Available{pos_str} Players:")
        click.echo(f"{'Name':<25} {'Pos':<3} {'Team':<4} {'Points':<7} {'VBD':<7} {'Tier'}")
        click.echo("-" * 55)
        
        for player in available:
            points = player.get('final_points', 0) or 0
            vbd = player.get('vbd_score', 0) or 0
            tier = player.get('tier', '-') or '-'
            
            click.echo(f"{player['name']:<25} {player['position']:<3} "
                      f"{player.get('team', 'N/A'):<4} {points:<7.1f} "
                      f"{vbd:<7.1f} {tier}")
    
    def _show_team_roster(self, draft_id: int, team_position: int, team_name: str):
        """Show team roster during draft"""
        import click
        
        roster = self.get_team_roster(draft_id, team_position)
        
        click.echo(f"\n👥 {team_name} Roster:")
        if not roster:
            click.echo("  No players drafted yet")
            return
            
        click.echo(f"{'Pick':<6} {'Slot':<8} {'Name':<20} {'Pos':<3} {'Points':<7}")
        click.echo("-" * 50)
        
        total_points = 0
        for player in roster:
            points = player.get('final_points', 0) or 0
            total_points += points
            
            click.echo(f"R{player['round_number']}P{player['pick_number']:<3} "
                      f"{player['position_slot']:<8} {player['name']:<20} "
                      f"{player['position']:<3} {points:<7.1f}")
        
        click.echo("-" * 50)
        click.echo(f"{'Total':<35} {total_points:<7.1f}")
    
    def _show_detailed_recommendations(self, draft_id: int, team_position: int):
        """Show detailed recommendations during draft"""
        import click
        
        recommendations = self.get_recommendations(draft_id, team_position)
        
        click.echo(f"\n🤖 Detailed Recommendations:")
        click.echo(f"{'#':<2} {'Name':<20} {'Pos':<3} {'Points':<7} {'VBD':<7} {'Need':<5} {'Reason'}")
        click.echo("-" * 75)
        
        for i, rec in enumerate(recommendations[:10], 1):
            click.echo(f"{i:<2} {rec['name']:<20} {rec['position']:<3} "
                      f"{rec.get('projected_points', 0):<7.1f} "
                      f"{rec['vbd_score']:<7.1f} "
                      f"{rec.get('positional_need', 0):<5.2f} "
                      f"{rec['reason']}")
    
    def _find_player_by_name(self, draft_id: int, name_input: str) -> Dict:
        """Find available player by partial name match"""
        available_players = self.get_available_players(draft_id, limit=500)
        
        # Exact match first
        for player in available_players:
            if player['name'].lower() == name_input.lower():
                return player
        
        # Partial match
        matches = [p for p in available_players if name_input.lower() in p['name'].lower()]
        
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            import click
            click.echo(f"Multiple matches found:")
            for i, match in enumerate(matches[:10], 1):
                points_str = f" ({match['final_points']:.1f})" if match.get('final_points') else ""
                click.echo(f"  {i}. {match['name']} ({match['position']}){points_str}")
            
            try:
                choice = click.prompt("Select number", type=int, default=1)
                if 1 <= choice <= len(matches):
                    return matches[choice - 1]
            except:
                pass
        
        return None
    
    def _show_final_results(self, draft_id: int):
        """Show final draft results"""
        import click
        
        draft_info = self.get_draft_info(draft_id)
        
        click.echo(f"\n📊 FINAL DRAFT RESULTS")
        click.echo(f"League: {draft_info['league_id']}")
        click.echo(f"Format: {draft_info['num_teams']} teams, {draft_info['total_rounds']} rounds")
        click.echo(f"Scoring: {draft_info['scoring_system'].upper()}")
        
        for team_pos in range(1, draft_info['num_teams'] + 1):
            team_name = draft_info['draft_order'][team_pos - 1]
            roster = self.get_team_roster(draft_id, team_pos)
            
            total_points = sum(player.get('final_points', 0) or 0 for player in roster)
            
            click.echo(f"\n{team_name} (Total: {total_points:.1f} pts):")
            for player in roster:
                points = player.get('final_points', 0) or 0
                click.echo(f"  R{player['round_number']:>2}: {player['position_slot']:<4} "
                          f"{player['name']:<20} ({player['position']}) {points:>5.1f}")
    
    def sync_sleeper_draft(self, league_id: str, sleeper_draft_info: Dict) -> int:
        """Sync draft state with Sleeper"""
        try:
            # Create or get existing draft
            existing_drafts = self.db.execute_query(
                "SELECT draft_id FROM drafts WHERE league_id = ?",
                (league_id,)
            )
            
            if existing_drafts:
                draft_id = existing_drafts[0]['draft_id']
                # Clear existing picks to resync
                self.db.execute_update("DELETE FROM draft_picks WHERE draft_id = ?", (draft_id,))
                self.db.execute_update("DELETE FROM team_rosters WHERE draft_id = ?", (draft_id,))
            else:
                # Create new draft
                draft_order = sleeper_draft_info.get('draft_order', [])
                draft_id = self.create_draft(
                    league_id, 
                    len(draft_order), 
                    'ppr',  # Default, will be updated from league settings
                    total_rounds=sleeper_draft_info.get('settings', {}).get('rounds', 16)
                )
            
            # Sync current picks
            picks = sleeper_draft_info.get('picks', [])
            for pick in picks:
                if pick.get('player_id'):
                    # Map Sleeper player ID to our player ID
                    player_id = self._map_sleeper_player_id(pick['player_id'])
                    if player_id:
                        self.make_pick(draft_id, player_id, pick['pick_no'])
            
            # Update draft status
            draft_status = sleeper_draft_info.get('status', 'active')
            if draft_status == 'complete':
                self.db.execute_update(
                    "UPDATE drafts SET status = 'completed' WHERE draft_id = ?",
                    (draft_id,)
                )
            
            return draft_id
            
        except Exception as e:
            raise Exception(f"Failed to sync Sleeper draft: {e}")
    
    def _map_sleeper_player_id(self, sleeper_player_id: str) -> str:
        """Map Sleeper player ID to internal player ID"""
        # This would require a mapping table or API call to get player info
        # For now, return None and implement based on actual Sleeper integration needs
        from integrations.sleeper import SleeperAPI
        
        try:
            sleeper_api = SleeperAPI()
            player_id = sleeper_api.map_sleeper_to_internal_player(sleeper_player_id)
            return player_id
        except:
            return None
    
    def monitor_sleeper_draft(self, draft_id: int, league_id: str, poll_interval: int = 30):
        """Monitor Sleeper draft for updates"""
        import click
        from integrations.sleeper import SleeperAPI
        
        sleeper_api = SleeperAPI()
        click.echo(f"👀 Monitoring Sleeper draft (polling every {poll_interval}s)...")
        click.echo("Press Ctrl+C to stop")
        
        last_pick_count = 0
        
        try:
            while True:
                # Get current draft state from Sleeper
                draft_info = sleeper_api.get_league_draft_info(league_id)
                
                if not draft_info:
                    click.echo("❌ Could not retrieve draft info")
                    break
                
                if draft_info.get('status') == 'complete':
                    click.echo("🏁 Draft completed!")
                    break
                
                # Check for new picks
                current_picks = draft_info.get('picks', [])
                new_pick_count = len(current_picks)
                
                if new_pick_count > last_pick_count:
                    # Sync new picks
                    self.sync_sleeper_draft(league_id, draft_info)
                    
                    # Show new picks
                    new_picks = current_picks[last_pick_count:]
                    for pick in new_picks:
                        player_name = pick.get('metadata', {}).get('full_name', 'Unknown Player')
                        team_name = f"Team {pick.get('picked_by', 'Unknown')}"
                        click.echo(f"📥 Pick {pick['pick_no']}: {player_name} → {team_name}")
                    
                    last_pick_count = new_pick_count
                
                # Show current recommendations if it's user's turn
                current_pick_info = self._get_current_pick_info(draft_id)
                if current_pick_info and current_pick_info.get('is_user_turn'):
                    recommendations = self.get_recommendations(draft_id)
                    if recommendations:
                        click.echo(f"\n🔔 YOUR TURN! Top recommendations:")
                        for i, rec in enumerate(recommendations[:3], 1):
                            click.echo(f"  {i}. {rec['name']} ({rec['position']}) - {rec['reason']}")
                
                time.sleep(poll_interval)
                
        except KeyboardInterrupt:
            click.echo("\n⏹️ Stopped monitoring draft")
        except Exception as e:
            click.echo(f"❌ Error monitoring draft: {e}")
    
    def _get_current_pick_info(self, draft_id: int) -> Dict:
        """Get information about current pick"""
        draft_info = self.get_draft_info(draft_id)
        if not draft_info:
            return None
        
        current_pick = draft_info['current_pick']
        total_picks = draft_info['total_rounds'] * draft_info['num_teams']
        
        if current_pick > total_picks:
            return {'is_complete': True}
        
        team_pos = self._get_team_position(current_pick, draft_info['num_teams'])
        current_team = draft_info['draft_order'][team_pos - 1]
        
        return {
            'pick_number': current_pick,
            'team_position': team_pos,
            'team_name': current_team,
            'is_user_turn': current_team == "USER",
            'is_complete': False
        }
    
    def get_draft_summary(self, draft_id: int) -> Dict:
        """Get comprehensive draft summary"""
        draft_info = self.get_draft_info(draft_id)
        if not draft_info:
            return None
        
        # Get all team rosters
        team_summaries = []
        for team_pos in range(1, draft_info['num_teams'] + 1):
            roster = self.get_team_roster(draft_id, team_pos)
            team_name = draft_info['draft_order'][team_pos - 1]
            total_points = sum(player.get('final_points', 0) or 0 for player in roster)
           
           # Calculate positional breakdown
            position_counts = {}
            for player in roster:
                pos = player['position']
                position_counts[pos] = position_counts.get(pos, 0) + 1
            
            team_summaries.append({
                'team_position': team_pos,
                'team_name': team_name,
                'roster': roster,
                'total_projected_points': total_points,
                'position_counts': position_counts,
                'roster_size': len(roster)
            })
        
        # Get draft picks summary
        total_picks = draft_info['total_rounds'] * draft_info['num_teams']
        picks_made = self.db.execute_query(
            "SELECT COUNT(*) as count FROM draft_picks WHERE draft_id = ?",
            (draft_id,)
        )[0]['count']
        
        return {
            'draft_info': draft_info,
            'team_summaries': team_summaries,
            'total_picks': total_picks,
            'picks_made': picks_made,
            'is_complete': picks_made >= total_picks,
            'completion_percentage': (picks_made / total_picks) * 100 if total_picks > 0 else 0
        }
    
    def undo_last_pick(self, draft_id: int) -> bool:
        """Undo the last pick in the draft"""
        try:
            # Get the last pick
            last_pick_query = """
            SELECT * FROM draft_picks 
            WHERE draft_id = ? 
            ORDER BY pick_number DESC 
            LIMIT 1
            """
            last_pick_result = self.db.execute_query(last_pick_query, (draft_id,))
            
            if not last_pick_result:
                return False
            
            last_pick = last_pick_result[0]
            
            # Remove from draft_picks
            self.db.execute_update(
                "DELETE FROM draft_picks WHERE draft_id = ? AND pick_number = ?",
                (draft_id, last_pick['pick_number'])
            )
            
            # Remove from team_rosters
            self.db.execute_update(
                "DELETE FROM team_rosters WHERE draft_id = ? AND player_id = ?",
                (draft_id, last_pick['player_id'])
            )
            
            # Update current pick
            self.db.execute_update(
                "UPDATE drafts SET current_pick = ?, status = 'active' WHERE draft_id = ?",
                (last_pick['pick_number'], draft_id)
            )
            
            return True
            
        except Exception as e:
            print(f"Error undoing pick: {e}")
            return False
    
    def get_position_scarcity(self, draft_id: int) -> Dict:
        """Analyze position scarcity based on remaining players"""
        positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']
        scarcity_data = {}
        
        for position in positions:
            available = self.get_available_players(draft_id, position, 100)
            
            # Get tiers remaining
            tier_counts = {}
            total_vbd = 0
            
            for player in available:
                tier = player.get('tier', 6)
                vbd = player.get('vbd_score', 0) or 0
                
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
                total_vbd += vbd
            
            # Calculate scarcity metrics
            top_tier_remaining = sum(tier_counts.get(tier, 0) for tier in [1, 2])
            total_remaining = len(available)
            avg_vbd = total_vbd / total_remaining if total_remaining > 0 else 0
            
            scarcity_data[position] = {
                'total_remaining': total_remaining,
                'top_tier_remaining': top_tier_remaining,
                'average_vbd': avg_vbd,
                'tier_breakdown': tier_counts,
                'scarcity_score': top_tier_remaining / total_remaining if total_remaining > 0 else 0
            }
        
        return scarcity_data
    
    def export_draft_results(self, draft_id: int, format: str = 'json') -> str:
        """Export draft results in specified format"""
        draft_summary = self.get_draft_summary(draft_id)
        
        if not draft_summary:
            raise ValueError("Draft not found")
        
        if format.lower() == 'json':
            import json
            return json.dumps(draft_summary, indent=2, default=str)
        
        elif format.lower() == 'csv':
            import csv
            from io import StringIO
            
            output = StringIO()
            writer = csv.writer(output)
            
            # Write headers
            writer.writerow(['Team', 'Pick', 'Round', 'Player', 'Position', 'Points', 'VBD'])
            
            # Write picks
            for team in draft_summary['team_summaries']:
                for player in team['roster']:
                    writer.writerow([
                        team['team_name'],
                        player['pick_number'],
                        player['round_number'],
                        player['name'],
                        player['position'],
                        player.get('final_points', 0),
                        player.get('vbd_score', 0)
                    ])
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def simulate_draft_pick(self, draft_id: int, player_id: str) -> Dict:
        """Simulate what would happen if a specific player is drafted"""
        try:
            draft_info = self.get_draft_info(draft_id)
            current_pick = draft_info['current_pick']
            team_pos = self._get_team_position(current_pick, draft_info['num_teams'])
            
            # Get player info
            player_info = self._get_player_info(player_id)
            if not player_info:
                return {'error': 'Player not found'}
            
            # Check if player is available
            if self._is_player_drafted(draft_id, player_id):
                return {'error': 'Player already drafted'}
            
            # Get current roster
            current_roster = self.get_team_roster(draft_id, team_pos)
            
            # Simulate position slot assignment
            position_slot = self._determine_position_slot(draft_id, team_pos, player_info['position'])
            
            # Get updated recommendations after this pick
            # (This would require temporarily making the pick and getting new recommendations)
            
            return {
                'player': player_info,
                'team_position': team_pos,
                'pick_number': current_pick,
                'position_slot': position_slot,
                'current_roster_size': len(current_roster),
                'projected_impact': {
                    'points_added': player_info.get('final_points', 0),
                    'vbd_added': player_info.get('vbd_score', 0),
                    'position_need_filled': player_info['position']
                }
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_team_needs_analysis(self, draft_id: int, team_position: int) -> Dict:
        """Analyze team's positional needs"""
        try:
            draft_info = self.get_draft_info(draft_id)
            roster_settings = draft_info['roster_settings']
            current_roster = self.get_team_roster(draft_id, team_position)
            
            # Count current positions
            current_positions = {}
            for player in current_roster:
                pos = player['position']
                current_positions[pos] = current_positions.get(pos, 0) + 1
            
            # Calculate needs
            needs_analysis = {}
            for pos, required in roster_settings.items():
                if pos == 'BENCH':
                    continue
                
                current = current_positions.get(pos, 0)
                
                if pos == 'FLEX':
                    # FLEX can be filled by RB/WR/TE
                    flex_eligible_positions = ['RB', 'WR', 'TE']
                    excess_skill_players = 0
                    
                    for flex_pos in flex_eligible_positions:
                        pos_required = roster_settings.get(flex_pos, 0)
                        pos_current = current_positions.get(flex_pos, 0)
                        if pos_current > pos_required:
                            excess_skill_players += pos_current - pos_required
                    
                    need_level = max(0, required - excess_skill_players)
                else:
                    need_level = max(0, required - current)
                
                # Calculate urgency based on remaining rounds
                total_picks = draft_info['total_rounds'] * draft_info['num_teams']
                remaining_picks = total_picks - draft_info['current_pick'] + 1
                remaining_rounds = remaining_picks // draft_info['num_teams']
                
                urgency = 'low'
                if need_level > 0:
                    if remaining_rounds <= need_level:
                        urgency = 'critical'
                    elif remaining_rounds <= need_level * 2:
                        urgency = 'high'
                    else:
                        urgency = 'medium'
                
                needs_analysis[pos] = {
                    'required': required,
                    'current': current,
                    'need_level': need_level,
                    'urgency': urgency,
                    'is_filled': need_level == 0
                }
            
            return needs_analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_draft_grades(self, draft_id: int) -> Dict:
        """Calculate draft grades for all teams"""
        try:
            draft_summary = self.get_draft_summary(draft_id)
            team_grades = {}
            
            # Get average points per team for comparison
            all_team_points = [team['total_projected_points'] for team in draft_summary['team_summaries']]
            avg_points = sum(all_team_points) / len(all_team_points) if all_team_points else 0
            
            for team in draft_summary['team_summaries']:
                team_points = team['total_projected_points']
                roster = team['roster']
                
                # Calculate various metrics
                roster_balance_score = self._calculate_roster_balance(team['position_counts'])
                value_score = sum(player.get('vbd_score', 0) or 0 for player in roster)
                
                # Normalize scores
                points_percentile = (team_points / avg_points) if avg_points > 0 else 1
                
                # Calculate overall grade (A-F scale)
                overall_score = (points_percentile * 0.4 + 
                                roster_balance_score * 0.3 + 
                                min(value_score / 50, 1.0) * 0.3)  # Normalize VBD
                
                if overall_score >= 0.9:
                    letter_grade = 'A+'
                elif overall_score >= 0.85:
                    letter_grade = 'A'
                elif overall_score >= 0.8:
                    letter_grade = 'A-'
                elif overall_score >= 0.75:
                    letter_grade = 'B+'
                elif overall_score >= 0.7:
                    letter_grade = 'B'
                elif overall_score >= 0.65:
                    letter_grade = 'B-'
                elif overall_score >= 0.6:
                    letter_grade = 'C+'
                elif overall_score >= 0.55:
                    letter_grade = 'C'
                elif overall_score >= 0.5:
                    letter_grade = 'C-'
                elif overall_score >= 0.45:
                    letter_grade = 'D+'
                elif overall_score >= 0.4:
                    letter_grade = 'D'
                else:
                    letter_grade = 'F'
                
                team_grades[team['team_name']] = {
                    'overall_grade': letter_grade,
                    'overall_score': overall_score,
                    'projected_points': team_points,
                    'total_vbd': value_score,
                    'roster_balance': roster_balance_score,
                    'strengths': self._identify_team_strengths(roster),
                    'weaknesses': self._identify_team_weaknesses(team['position_counts'])
                }
            
            return team_grades
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_roster_balance(self, position_counts: Dict) -> float:
        """Calculate how balanced a roster is (0-1 scale)"""
        ideal_distribution = {'QB': 1, 'RB': 3, 'WR': 4, 'TE': 2, 'K': 1, 'DEF': 1}
        
        balance_score = 0
        total_positions = len(ideal_distribution)
        
        for pos, ideal_count in ideal_distribution.items():
            actual_count = position_counts.get(pos, 0)
            # Penalize both shortages and excesses, but shortages more severely
            if actual_count < ideal_count:
                position_score = actual_count / ideal_count
            else:
                excess = actual_count - ideal_count
                position_score = max(0.5, 1 - (excess * 0.1))  # Diminishing returns for excess
            
            balance_score += position_score
        
        return balance_score / total_positions
    
    def _identify_team_strengths(self, roster: List[Dict]) -> List[str]:
        """Identify team strengths based on roster"""
        strengths = []
        
        # Group by position
        by_position = {}
        for player in roster:
            pos = player['position']
            if pos not in by_position:
                by_position[pos] = []
            by_position[pos].append(player)
        
        # Check for strong position groups
        for pos, players in by_position.items():
            if len(players) >= 2:  # Multiple players at position
                avg_vbd = sum(p.get('vbd_score', 0) or 0 for p in players) / len(players)
                if avg_vbd > 10:  # Strong VBD average
                    strengths.append(f"Strong {pos} corps")
            
            # Check for elite individual players
            for player in players:
                if player.get('tier', 6) <= 2:  # Top tier player
                    strengths.append(f"Elite {pos}: {player['name']}")
        
        return strengths[:3]  # Limit to top 3 strengths
    
    def _identify_team_weaknesses(self, position_counts: Dict) -> List[str]:
        """Identify team weaknesses based on position counts"""
        weaknesses = []
        minimum_needs = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1}
        
        for pos, min_needed in minimum_needs.items():
            current = position_counts.get(pos, 0)
            if current < min_needed:
                if current == 0:
                    weaknesses.append(f"No {pos} drafted")
                else:
                    weaknesses.append(f"Thin at {pos} ({current} players)")
        
        return weaknesses