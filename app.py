from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import json
import logging
from datetime import datetime
import traceback
import os
import sys

# Import your existing ML system
try:
    from ml_draft_analyzer import AdvancedMLDraftSystem
except ImportError:
    print("Error: Could not import AdvancedMLDraftSystem. Make sure the file is in the same directory.")
    sys.exit(1)

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global draft system instance
draft_system = None

def initialize_draft_system():
    """Initialize the draft system"""
    global draft_system
    try:
        draft_system = AdvancedMLDraftSystem()
        logger.info("Draft system initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize draft system: {e}")
        return False

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    """Get system status"""
    global draft_system
    
    models_trained = len(draft_system.models) > 0 if draft_system else False
    projections_ready = len(draft_system.available_players) > 0 if draft_system else False
    
    return jsonify({
        'system_initialized': draft_system is not None,
        'models_trained': models_trained,
        'projections_ready': projections_ready,
        'league_settings': draft_system.league_settings if draft_system else {}
    })

@app.route('/api/train-models', methods=['POST'])
def api_train_models():
    """Train ML models"""
    global draft_system
    
    try:
        if not draft_system:
            return jsonify({'error': 'System not initialized'}), 500
        
        # Load training data
        training_data = draft_system.load_comprehensive_training_data()
        
        if training_data.empty:
            return jsonify({'error': 'No training data found'}), 400
        
        # Engineer features
        training_data = draft_system.engineer_advanced_features(training_data)
        
        # Train models
        results = draft_system.train_advanced_models(training_data)
        
        return jsonify({
            'success': True,
            'message': f'Models trained for {len(results)} positions',
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error training models: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-projections', methods=['POST'])
def api_generate_projections():
    """Generate 2025 player projections"""
    global draft_system
    
    try:
        if not draft_system:
            return jsonify({'error': 'System not initialized'}), 500
        
        # Ensure models are trained
        if not draft_system.models:
            return jsonify({'error': 'Models not trained. Train models first.'}), 400
        
        # Generate projections
        projections = draft_system.get_2025_player_projections()
        
        if not projections:
            return jsonify({'error': 'Failed to generate projections'}), 500
        
        # Calculate VBD and get rankings
        rankings = draft_system.calculate_proper_vbd(projections)
        
        # Add tiers and grades
        for i, player in enumerate(rankings):
            player['overall_rank'] = i + 1
            player['tier'] = draft_system._assign_tier(player['vbd_score'])
            player['draft_grade'] = draft_system._assign_draft_grade(player['vbd_score'], player['position'])
        
        # Store in system
        draft_system.available_players = rankings
        
        return jsonify({
            'success': True,
            'message': f'Generated projections for {len(rankings)} players',
            'player_count': len(rankings)
        })
        
    except Exception as e:
        logger.error(f"Error generating projections: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rankings')
def api_rankings():
    """Get player rankings"""
    global draft_system
    
    try:
        if not draft_system or not draft_system.available_players:
            return jsonify({'error': 'Projections not generated'}), 400
        
        # Get pagination parameters
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))
        position = request.args.get('position', '').upper()
        
        # Filter by position if specified
        players = draft_system.available_players
        if position and position in ['QB', 'RB', 'WR', 'TE']:
            players = [p for p in players if p['position'] == position]
        
        # Paginate
        start = (page - 1) * per_page
        end = start + per_page
        page_players = players[start:end]
        
        return jsonify({
            'players': page_players,
            'total': len(players),
            'page': page,
            'per_page': per_page,
            'has_next': end < len(players),
            'has_prev': page > 1
        })
        
    except Exception as e:
        logger.error(f"Error getting rankings: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/setup-draft', methods=['POST'])
def api_setup_draft():
    """Setup draft simulation"""
    global draft_system
    
    try:
        if not draft_system:
            return jsonify({'error': 'System not initialized'}), 500
        
        data = request.get_json()
        
        # Update league settings
        teams = int(data.get('teams', 12))
        my_position = int(data.get('my_position', 1))
        rounds = int(data.get('rounds', 16))
        scoring = data.get('scoring', 'ppr')
        
        draft_system.league_settings.update({
            'teams': teams,
            'rounds': rounds,
            'scoring': scoring
        })
        
        draft_system.my_pick_position = my_position
        
        # Initialize draft state
        draft_system._generate_draft_order()
        draft_system.current_round = 1
        draft_system.current_pick = 1
        draft_system.drafted_players = set()
        draft_system.my_roster = {
            'QB': [], 'RB': [], 'WR': [], 'TE': [], 'K': [], 'DEF': [], 'BENCH': []
        }
        
        # Initialize all team rosters
        draft_system.all_team_rosters = {}
        for team_num in range(1, teams + 1):
            draft_system.all_team_rosters[team_num] = {
                'QB': [], 'RB': [], 'WR': [], 'TE': [], 'K': [], 'DEF': [], 'BENCH': []
            }
        
        # Store draft settings in session
        session['draft_active'] = True
        session['teams'] = teams
        session['my_position'] = my_position
        session['rounds'] = rounds
        
        return jsonify({
            'success': True,
            'message': 'Draft setup complete',
            'next_picks': draft_system._get_my_upcoming_picks()[:5]
        })
        
    except Exception as e:
        logger.error(f"Error setting up draft: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/draft-status')
def api_draft_status():
    """Get current draft status"""
    global draft_system
    
    try:
        if not draft_system or not session.get('draft_active'):
            return jsonify({'error': 'No active draft'}), 400
        
        # Get current pick info
        current_pick_info = {}
        if draft_system.current_pick <= len(draft_system.draft_order):
            current_pick_info = draft_system.draft_order[draft_system.current_pick - 1]
        
        # Get team analysis
        analysis = draft_system.analyze_team_composition()
        
        # Get my upcoming picks
        my_picks = draft_system._get_my_upcoming_picks()
        current_pick_idx = next((i for i, pick in enumerate(my_picks) if pick >= draft_system.current_pick), None)
        upcoming_picks = my_picks[current_pick_idx:current_pick_idx+3] if current_pick_idx is not None else []
        
        return jsonify({
            'current_pick': draft_system.current_pick,
            'current_round': draft_system.current_round,
            'is_my_pick': current_pick_info.get('is_my_pick', False),
            'current_team': current_pick_info.get('team_position'),
            'my_roster': draft_system.my_roster,
            'upcoming_picks': upcoming_picks,
            'team_analysis': analysis,
            'total_picks': len(draft_system.draft_order),
            'draft_complete': draft_system.current_pick > len(draft_system.draft_order)
        })
        
    except Exception as e:
        logger.error(f"Error getting draft status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/draft-recommendations')
def api_draft_recommendations():
    """Get draft recommendations"""
    global draft_system
    
    try:
        if not draft_system or not session.get('draft_active'):
            return jsonify({'error': 'No active draft'}), 400
        
        top_n = int(request.args.get('count', 8))
        recommendations = draft_system.get_draft_recommendations(top_n)
        
        return jsonify({
            'recommendations': recommendations
        })
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/draft-player', methods=['POST'])
def api_draft_player():
    """Draft a player"""
    global draft_system
    
    try:
        if not draft_system or not session.get('draft_active'):
            return jsonify({'error': 'No active draft'}), 400
        
        data = request.get_json()
        player_id = data.get('player_id')
        
        if not player_id:
            return jsonify({'error': 'Player ID required'}), 400
        
        # Find player in available players
        player = next((p for p in draft_system.available_players if p['player_id'] == player_id), None)
        
        if not player:
            return jsonify({'error': 'Player not found or already drafted'}), 400
        
        # Check if it's actually the user's pick
        current_pick_info = draft_system.draft_order[draft_system.current_pick - 1]
        if not current_pick_info['is_my_pick']:
            return jsonify({'error': 'Not your pick'}), 400
        
        # Draft the player
        draft_system._draft_player(player, is_my_pick=True)
        
        return jsonify({
            'success': True,
            'message': f'Drafted {player["name"]}',
            'player': player
        })
        
    except Exception as e:
        logger.error(f"Error drafting player: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/simulate-cpu-pick', methods=['POST'])
def api_simulate_cpu_pick():
    """Simulate CPU pick"""
    global draft_system
    
    try:
        if not draft_system or not session.get('draft_active'):
            return jsonify({'error': 'No active draft'}), 400
        
        current_pick_info = draft_system.draft_order[draft_system.current_pick - 1]
        
        if current_pick_info['is_my_pick']:
            return jsonify({'error': 'This is your pick, not CPU'}), 400
        
        # Simulate CPU pick
        drafted_player = draft_system.simulate_cpu_pick(current_pick_info['team_position'])
        
        if drafted_player:
            return jsonify({
                'success': True,
                'player': drafted_player,
                'team': current_pick_info['team_position']
            })
        else:
            return jsonify({'error': 'Failed to simulate CPU pick'}), 500
        
    except Exception as e:
        logger.error(f"Error simulating CPU pick: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/available-players')
def api_available_players():
    """Get available players for drafting"""
    global draft_system
    
    try:
        if not draft_system:
            return jsonify({'error': 'System not initialized'}), 400
        
        position = request.args.get('position', '').upper()
        search = request.args.get('search', '').lower()
        limit = int(request.args.get('limit', 50))
        
        # Filter available players
        players = draft_system.available_players or []
        
        if position and position in ['QB', 'RB', 'WR', 'TE']:
            players = [p for p in players if p['position'] == position]
        
        if search:
            players = [p for p in players if search in p['name'].lower()]
        
        # Limit results
        players = players[:limit]
        
        return jsonify({
            'players': players,
            'total_available': len(draft_system.available_players) if draft_system.available_players else 0
        })
        
    except Exception as e:
        logger.error(f"Error getting available players: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/league-settings', methods=['GET', 'POST'])
def api_league_settings():
    """Get or update league settings"""
    global draft_system
    
    if request.method == 'GET':
        return jsonify({
            'settings': draft_system.league_settings if draft_system else {}
        })
    
    elif request.method == 'POST':
        try:
            if not draft_system:
                return jsonify({'error': 'System not initialized'}), 500
            
            data = request.get_json()
            
            # Update settings
            if 'teams' in data:
                draft_system.league_settings['teams'] = int(data['teams'])
            if 'scoring' in data:
                draft_system.league_settings['scoring'] = data['scoring']
            if 'roster_spots' in data:
                draft_system.league_settings['roster_spots'] = data['roster_spots']
            
            return jsonify({
                'success': True,
                'settings': draft_system.league_settings
            })
            
        except Exception as e:
            logger.error(f"Error updating settings: {e}")
            return jsonify({'error': str(e)}), 500
# Add these new routes to your app.py file

@app.route('/api/manual-draft-pick', methods=['POST'])
def api_manual_draft_pick():
    """Manually enter a draft pick for another team"""
    global draft_system
    
    try:
        if not draft_system or not session.get('draft_active'):
            return jsonify({'error': 'No active draft'}), 400
        
        data = request.get_json()
        player_name = data.get('player_name', '').strip()
        team_position = data.get('team_position')
        
        if not player_name:
            return jsonify({'error': 'Player name required'}), 400
        
        # Find player by name (fuzzy matching)
        player = None
        for p in draft_system.available_players:
            if player_name.lower() in p['name'].lower() or p['name'].lower() in player_name.lower():
                player = p
                break
        
        if not player:
            return jsonify({'error': f'Player "{player_name}" not found in available players'}), 400
        
        # Verify it's not the user's pick
        current_pick_info = draft_system.draft_order[draft_system.current_pick - 1]
        if current_pick_info['is_my_pick']:
            return jsonify({'error': 'This is your pick, not a manual entry'}), 400
        
        # Use provided team position or current team
        team_pos = team_position or current_pick_info['team_position']
        
        # Draft the player
        draft_system._draft_player(player, team_position=team_pos)
        
        return jsonify({
            'success': True,
            'message': f'Team {team_pos} drafted {player["name"]}',
            'player': player
        })
        
    except Exception as e:
        logger.error(f"Error with manual draft pick: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/sleeper-sync', methods=['POST'])
def api_sleeper_sync():
    """Sync draft state with Sleeper API"""
    global draft_system
    
    try:
        if not draft_system:
            return jsonify({'error': 'System not initialized'}), 500
        
        data = request.get_json()
        draft_id = data.get('draft_id')
        
        if not draft_id:
            return jsonify({'error': 'Sleeper draft ID required'}), 400
        
        # Get draft picks from Sleeper
        sleeper_picks = fetch_sleeper_draft_picks(draft_id)
        
        if not sleeper_picks:
            return jsonify({'error': 'Failed to fetch Sleeper draft data'}), 500
        
        # Sync the picks with our system
        synced_count = sync_sleeper_picks(sleeper_picks)
        
        return jsonify({
            'success': True,
            'message': f'Synced {synced_count} picks from Sleeper',
            'total_picks': len(sleeper_picks)
        })
        
    except Exception as e:
        logger.error(f"Error syncing with Sleeper: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/sleeper-setup', methods=['POST'])
def api_sleeper_setup():
    """Setup draft using Sleeper league information"""
    global draft_system
    
    try:
        data = request.get_json()
        league_id = data.get('league_id')
        user_id = data.get('user_id')  # Your Sleeper user ID
        
        if not league_id:
            return jsonify({'error': 'Sleeper league ID required'}), 400
        
        # Fetch league and draft info from Sleeper
        league_info = fetch_sleeper_league_info(league_id)
        draft_info = fetch_sleeper_draft_info(league_id)
        
        if not league_info or not draft_info:
            return jsonify({'error': 'Failed to fetch Sleeper league data'}), 500
        
        # Setup draft system with Sleeper data
        setup_success = setup_draft_from_sleeper(league_info, draft_info, user_id)
        
        if setup_success:
            session['draft_active'] = True
            session['sleeper_draft_id'] = draft_info.get('draft_id')
            session['sleeper_league_id'] = league_id
            
            return jsonify({
                'success': True,
                'message': 'Draft setup with Sleeper data',
                'league_name': league_info.get('name'),
                'draft_type': draft_info.get('type'),
                'teams': len(league_info.get('roster_positions', []))
            })
        else:
            return jsonify({'error': 'Failed to setup draft with Sleeper data'}), 500
        
    except Exception as e:
        logger.error(f"Error setting up Sleeper draft: {e}")
        return jsonify({'error': str(e)}), 500


def fetch_sleeper_draft_picks(draft_id):
    """Fetch draft picks from Sleeper API"""
    import requests
    
    try:
        url = f"https://api.sleeper.app/v1/draft/{draft_id}/picks"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Sleeper API error: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Error fetching Sleeper picks: {e}")
        return None


def fetch_sleeper_league_info(league_id):
    """Fetch league information from Sleeper API"""
    import requests
    
    try:
        url = f"https://api.sleeper.app/v1/league/{league_id}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Sleeper API error: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Error fetching Sleeper league: {e}")
        return None


def fetch_sleeper_draft_info(league_id):
    """Fetch draft information from Sleeper API"""
    import requests
    
    try:
        # Get drafts for the league
        url = f"https://api.sleeper.app/v1/league/{league_id}/drafts"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            drafts = response.json()
            # Return the most recent draft
            if drafts:
                return drafts[0]
        
        logger.error(f"No drafts found for league {league_id}")
        return None
            
    except Exception as e:
        logger.error(f"Error fetching Sleeper draft info: {e}")
        return None


def setup_draft_from_sleeper(league_info, draft_info, user_id):
    """Setup draft system using Sleeper league data"""
    global draft_system
    
    try:
        # Extract league settings
        teams = len(league_info.get('rosters', []))
        scoring = 'ppr' if league_info.get('scoring_settings', {}).get('rec', 0) == 1 else 'std'
        
        # Find user's draft position
        draft_order = draft_info.get('draft_order')
        user_position = None
        
        if draft_order and user_id:
            for pos, sleeper_user_id in draft_order.items():
                if sleeper_user_id == user_id:
                    user_position = int(pos)
                    break
        
        if not user_position:
            user_position = 1  # Default if not found
        
        # Update draft system settings
        draft_system.league_settings.update({
            'teams': teams,
            'rounds': 16,  # Standard fantasy rounds
            'scoring': scoring
        })
        
        draft_system.my_pick_position = user_position
        
        # Initialize draft state
        draft_system._generate_draft_order()
        draft_system.current_round = 1
        draft_system.current_pick = 1
        draft_system.drafted_players = set()
        draft_system.my_roster = {
            'QB': [], 'RB': [], 'WR': [], 'TE': [], 'K': [], 'DEF': [], 'BENCH': []
        }
        
        # Initialize all team rosters
        draft_system.all_team_rosters = {}
        for team_num in range(1, teams + 1):
            draft_system.all_team_rosters[team_num] = {
                'QB': [], 'RB': [], 'WR': [], 'TE': [], 'K': [], 'DEF': [], 'BENCH': []
            }
        
        return True
        
    except Exception as e:
        logger.error(f"Error setting up draft from Sleeper: {e}")
        return False


def sync_sleeper_picks(sleeper_picks):
    """Sync Sleeper draft picks with our system"""
    global draft_system
    
    synced_count = 0
    
    try:
        # Sort picks by pick number
        sorted_picks = sorted(sleeper_picks, key=lambda x: x.get('pick_no', 0))
        
        for pick_data in sorted_picks:
            player_id = pick_data.get('player_id')
            pick_no = pick_data.get('pick_no', 0)
            
            if not player_id or pick_no <= 0:
                continue
            
            # Find player in our system by Sleeper player ID
            our_player = find_player_by_sleeper_id(player_id)
            
            if not our_player:
                # Try to find by name if we have it
                metadata = pick_data.get('metadata', {})
                if metadata:
                    player_name = f"{metadata.get('first_name', '')} {metadata.get('last_name', '')}".strip()
                    our_player = find_player_by_name(player_name)
            
            if our_player and our_player['player_id'] not in draft_system.drafted_players:
                # Calculate which team made this pick
                teams = draft_system.league_settings['teams']
                round_num = ((pick_no - 1) // teams) + 1
                pick_in_round = ((pick_no - 1) % teams) + 1
                
                # Account for snake draft
                if round_num % 2 == 0:  # Even rounds are reversed
                    team_position = teams - pick_in_round + 1
                else:  # Odd rounds are normal
                    team_position = pick_in_round
                
                # Update our draft state to this pick
                draft_system.current_pick = pick_no
                draft_system.current_round = round_num
                
                # Draft the player
                is_my_pick = team_position == draft_system.my_pick_position
                draft_system._draft_player(our_player, team_position, is_my_pick)
                
                synced_count += 1
        
        return synced_count
        
    except Exception as e:
        logger.error(f"Error syncing Sleeper picks: {e}")
        return 0


def find_player_by_sleeper_id(sleeper_player_id):
    """Find player in our system by Sleeper player ID"""
    global draft_system
    
    # This would require mapping Sleeper player IDs to our player IDs
    # For now, we'll return None and fall back to name matching
    # In a full implementation, you'd maintain a mapping table
    return None


def find_player_by_name(player_name):
    """Find player in our system by name"""
    global draft_system
    
    if not player_name or not draft_system.available_players:
        return None
    
    player_name_lower = player_name.lower()
    
    # Try exact match first
    for player in draft_system.available_players:
        if player['name'].lower() == player_name_lower:
            return player
    
    # Try partial match
    for player in draft_system.available_players:
        if player_name_lower in player['name'].lower() or player['name'].lower() in player_name_lower:
            return player
    
    return None


@app.route('/api/undo-pick', methods=['POST'])
def api_undo_pick():
    """Undo the last draft pick"""
    global draft_system
    
    try:
        if not draft_system or not session.get('draft_active'):
            return jsonify({'error': 'No active draft'}), 400
        
        if draft_system.current_pick <= 1:
            return jsonify({'error': 'No picks to undo'}), 400
        
        # This is a simplified undo - in a full implementation you'd need
        # to track pick history and properly restore state
        draft_system.current_pick -= 1
        draft_system.current_round = ((draft_system.current_pick - 1) // draft_system.league_settings['teams']) + 1
        
        return jsonify({
            'success': True,
            'message': f'Undid pick #{draft_system.current_pick + 1}'
        })
        
    except Exception as e:
        logger.error(f"Error undoing pick: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/draft-mode', methods=['POST'])
def api_set_draft_mode():
    """Set draft input mode (manual vs sleeper vs simulation)"""
    global draft_system
    
    try:
        data = request.get_json()
        mode = data.get('mode', 'simulation')  # simulation, manual, sleeper
        
        if mode not in ['simulation', 'manual', 'sleeper']:
            return jsonify({'error': 'Invalid mode. Use: simulation, manual, or sleeper'}), 400
        
        session['draft_mode'] = mode
        
        return jsonify({
            'success': True,
            'mode': mode,
            'message': f'Draft mode set to {mode}'
        })
        
    except Exception as e:
        logger.error(f"Error setting draft mode: {e}")
        return jsonify({'error': str(e)}), 500


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error='Page not found'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error='Internal server error'), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🏈 Fantasy Football ML Draft System - Web Interface")
    print("=" * 60)
    
    # Initialize the draft system
    print("Initializing draft system...")
    if initialize_draft_system():
        print("✅ Draft system initialized successfully")
        print("\n🌐 Starting web server...")
        print("📍 Open your browser to: http://localhost:5000")
        print("⚠️ Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Run the Flask app
        app.run(debug=True, host='localhost', port=5000)
    else:
        print("❌ Failed to initialize draft system")
        print("Make sure the database and ML system files are properly set up.")