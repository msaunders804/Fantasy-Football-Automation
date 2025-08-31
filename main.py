#!/usr/bin/env python3
"""
Fantasy Football Draft Assistant - CLI Interface
"""
import click
from core.database import DatabaseManager
from projections.manager import ProjectionManager
from draft.engine import DraftEngine
from integrations.sleeper import SleeperAPI

@click.group()
@click.pass_context
def cli(ctx):
    """Fantasy Football Draft Assistant"""
    ctx.ensure_object(dict)
    ctx.obj['db'] = DatabaseManager()
    ctx.obj['projection_manager'] = ProjectionManager(ctx.obj['db'])
    ctx.obj['draft_engine'] = DraftEngine(ctx.obj['db'])

@cli.group()
def projections():
    """Projection management commands"""
    pass

@projections.command()
@click.option('--source', required=True, help='Projection source name')
@click.option('--file', 'file_path', required=True, help='CSV/JSON file path')
@click.option('--format', default='csv', help='File format (csv/json)')
@click.pass_context
def import_projections(ctx, source, file_path, format):
    """Import projections from file"""
    try:
        count = ctx.obj['projection_manager'].import_from_file(source, file_path, format)
        click.echo(f"Successfully imported {count} projections from {source}")
    except Exception as e:
        click.echo(f"Error importing projections: {e}", err=True)

@projections.command()
@click.option('--api-key', required=True, help='FantasyPros API key')
@click.option('--scoring', default='PPR', help='Scoring format (PPR, HALF, STD)')
@click.option('--week', default=0, help='Week number (0 for season)')
@click.pass_context
def import_fantasypros_api(ctx, api_key, scoring, week):
    """Import projections from FantasyPros API"""
    try:
        count = ctx.obj['projection_manager'].import_from_fantasypros_api(api_key, week, scoring)
        click.echo(f"Successfully imported {count} projections from FantasyPros API ({scoring})")
    except Exception as e:
        click.echo(f"Error importing from FantasyPros API: {e}", err=True)

@projections.command()
@click.option('--file', 'file_path', required=True, help='ESPN CSV file path')
@click.pass_context
def import_espn_csv(ctx, file_path):
    """Import projections from ESPN CSV"""
    try:
        count = ctx.obj['projection_manager'].import_from_espn_csv(file_path)
        click.echo(f"Successfully imported {count} projections from ESPN CSV")
    except Exception as e:
        click.echo(f"Error importing ESPN CSV: {e}", err=True)

@projections.command()
@click.option('--fantasypros-api-key', help='FantasyPros API key')
@click.option('--fantasypros-scoring', default='PPR', help='FantasyPros scoring format')
@click.option('--espn-csv', help='Path to ESPN CSV file')
@click.option('--additional-csv', help='Path to additional CSV file')
@click.option('--additional-source-name', default='Additional', help='Name for additional source')
@click.option('--auto-aggregate', is_flag=True, default=True, help='Automatically run aggregation after import')
@click.pass_context
def import_multiple(ctx, fantasypros_api_key, fantasypros_scoring, espn_csv, 
                   additional_csv, additional_source_name, auto_aggregate):
    """Import projections from multiple sources at once"""
    total_count = 0
    sources_imported = []
    
    try:
        if fantasypros_api_key:
            click.echo("📡 Importing from FantasyPros API...")
            count = ctx.obj['projection_manager'].import_from_fantasypros_api(
                fantasypros_api_key, 0, fantasypros_scoring
            )
            total_count += count
            sources_imported.append(f"FantasyPros API ({fantasypros_scoring})")
            click.echo(f"  ✅ {count} projections imported from FantasyPros API")
        
        if espn_csv:
            click.echo("📊 Importing from ESPN CSV...")
            count = ctx.obj['projection_manager'].import_from_espn_csv(espn_csv)
            total_count += count
            sources_imported.append("ESPN CSV")
            click.echo(f"  ✅ {count} projections imported from ESPN CSV")
        
        if additional_csv:
            click.echo(f"📋 Importing from {additional_source_name} CSV...")
            count = ctx.obj['projection_manager'].import_from_file(
                additional_source_name, additional_csv, 'csv'
            )
            total_count += count
            sources_imported.append(additional_source_name)
            click.echo(f"  ✅ {count} projections imported from {additional_source_name}")
        
        if total_count > 0:
            click.echo(f"\n📈 Total: {total_count} projections imported from {len(sources_imported)} sources")
            click.echo(f"Sources: {', '.join(sources_imported)}")
            
            if auto_aggregate:
                click.echo("\n🤖 Running ML aggregation...")
                success = ctx.obj['projection_manager'].aggregate_projections()
                if success:
                    click.echo("✅ Aggregation complete! Ready for draft.")
                else:
                    click.echo("⚠️ Aggregation had issues. Check logs.")
        else:
            click.echo("❌ No sources specified. Use --help for options.")
            click.echo("\nExample usage:")
            click.echo("  python main.py projections import-multiple \\")
            click.echo("    --fantasypros-api-key YOUR_KEY \\")
            click.echo("    --espn-csv projections.csv")
            
    except Exception as e:
        click.echo(f"❌ Error in multi-import: {e}", err=True)

@projections.command()
@click.pass_context
def aggregate(ctx):
    """Run ML aggregation on all projections"""
    try:
        click.echo("🤖 Running ML aggregation on projections...")
        success = ctx.obj['projection_manager'].aggregate_projections()
        if success:
            click.echo("✅ Projections aggregated successfully")
        else:
            click.echo("⚠️ Aggregation completed with warnings")
    except Exception as e:
        click.echo(f"❌ Error aggregating projections: {e}", err=True)

@projections.command()
@click.option('--position', help='Filter by position (QB, RB, WR, TE, K, DEF)')
@click.option('--limit', default=20, help='Number of players to show')
@click.option('--min-tier', type=int, help='Show only players in tier X or better')
@click.option('--show-details', is_flag=True, help='Show detailed stats')
@click.pass_context
def rankings(ctx, position, limit, min_tier, show_details):
    """Show current player rankings"""
    try:
        rankings = ctx.obj['projection_manager'].get_rankings(position, limit * 2)  # Get extra for filtering
        
        # Filter by tier if specified
        if min_tier:
            rankings = [p for p in rankings if p.get('tier', 99) <= min_tier]
        
        # Limit results
        rankings = rankings[:limit]
        
        if not rankings:
            click.echo("No players found matching criteria")
            return
            
        click.echo(f"\n🏆 {'Position-Specific' if position else 'Overall'} Rankings")
        click.echo(f"📊 Showing top {len(rankings)} players{' for ' + position if position else ''}")
        
        if show_details:
            click.echo(f"{'Rank':<4} {'Name':<25} {'Pos':<3} {'Team':<4} {'Points':<7} {'VBD':<7} {'Tier':<4} {'Conf':<4}")
            click.echo("─" * 75)
            for i, player in enumerate(rankings, 1):
                click.echo(f"{i:<4} {player['name']:<25} {player['position']:<3} "
                          f"{player.get('team', 'N/A'):<4} {player['points']:<7.1f} "
                          f"{player['vbd_score']:<7.1f} {player['tier']:<4} "
                          f"{player.get('confidence_score', 0):<4.2f}")
        else:
            click.echo(f"{'Rank':<4} {'Name':<25} {'Pos':<3} {'Points':<7} {'VBD':<7} {'Tier':<4}")
            click.echo("─" * 60)
            for i, player in enumerate(rankings, 1):
                click.echo(f"{i:<4} {player['name']:<25} {player['position']:<3} "
                          f"{player['points']:<7.1f} {player['vbd_score']:<7.1f} {player['tier']:<4}")
                          
    except Exception as e:
        click.echo(f"❌ Error getting rankings: {e}", err=True)

@projections.command()
@click.pass_context
def sources(ctx):
    """Show projection sources and their weights"""
    try:
        query = """
        SELECT ps.name, ps.weight, COUNT(p.projection_id) as projection_count,
               ps.accuracy_score, ps.created_at
        FROM projection_sources ps
        LEFT JOIN projections p ON ps.source_id = p.source_id
        GROUP BY ps.source_id, ps.name, ps.weight, ps.accuracy_score
        ORDER BY ps.weight DESC
        """
        
        sources = ctx.obj['db'].execute_query(query)
        
        if not sources:
            click.echo("No projection sources found")
            return
        
        click.echo(f"\n📊 Projection Sources")
        click.echo(f"{'Source':<20} {'Weight':<8} {'Count':<8} {'Accuracy':<10} {'Created':<12}")
        click.echo("─" * 70)
        
        for source in sources:
            created_date = source['created_at'][:10] if source['created_at'] else 'N/A'
            accuracy = f"{source['accuracy_score']:.3f}" if source['accuracy_score'] else 'N/A'
            
            click.echo(f"{source['name']:<20} {source['weight']:<8.2f} "
                      f"{source['projection_count']:<8} {accuracy:<10} {created_date:<12}")
        
        total_projections = sum(s['projection_count'] for s in sources)
        click.echo(f"\nTotal projections: {total_projections}")
        
    except Exception as e:
        click.echo(f"❌ Error getting sources: {e}", err=True)

@cli.group()
def draft():
    """Draft management commands"""
    pass

@draft.command()
@click.option('--league-id', required=True, help='League ID')
@click.option('--teams', default=12, help='Number of teams')
@click.option('--scoring', default='ppr', help='Scoring system (ppr/half_ppr/standard)')
@click.option('--position', type=int, help='Your draft position (1-N)')
@click.option('--rounds', default=16, help='Number of draft rounds')
@click.pass_context
def start(ctx, league_id, teams, scoring, position, rounds):
    """Start a new manual draft"""
    try:
        click.echo(f"🏈 Starting new draft...")
        click.echo(f"League: {league_id}")
        click.echo(f"Teams: {teams}, Rounds: {rounds}, Scoring: {scoring.upper()}")
        if position:
            click.echo(f"Your draft position: {position}")
        
        draft_id = ctx.obj['draft_engine'].create_draft(league_id, teams, scoring, position, rounds)
        click.echo(f"✅ Draft created! Draft ID: {draft_id}")
        
        # Check if we have projections
        rankings = ctx.obj['projection_manager'].get_rankings(limit=10)
        if not rankings:
            click.echo("⚠️ Warning: No aggregated projections found!")
            click.echo("Run 'python main.py projections aggregate' first for better recommendations")
            
            if not click.confirm("Continue without projections?"):
                return
        
        click.echo(f"\n🚀 Entering draft mode...")
        click.echo("💡 Commands: 'help', 'available [POS]', 'quit'")
        
        # Enter draft loop
        ctx.obj['draft_engine'].run_draft_loop(draft_id)
        
    except Exception as e:
        click.echo(f"❌ Error starting draft: {e}", err=True)

@draft.command()
@click.option('--league-id', required=True, help='Sleeper league ID')
@click.option('--poll-interval', default=30, help='Polling interval in seconds')
@click.pass_context
def sleeper(ctx, league_id, poll_interval):
    """Connect to Sleeper league and monitor draft"""
    try:
        click.echo(f"🔗 Connecting to Sleeper league: {league_id}")
        
        sleeper_api = SleeperAPI()
        draft_info = sleeper_api.get_league_draft_info(league_id)
        
        if not draft_info:
            click.echo("❌ No active draft found for this league")
            click.echo("Make sure the league ID is correct and draft is active")
            return
        
        click.echo(f"✅ Found draft: {draft_info.get('type', 'Unknown')} format")
        click.echo(f"Status: {draft_info.get('status', 'Unknown')}")
        
        # Sync draft state and start monitoring
        draft_id = ctx.obj['draft_engine'].sync_sleeper_draft(league_id, draft_info)
        click.echo(f"📊 Synced to internal draft ID: {draft_id}")
        
        ctx.obj['draft_engine'].monitor_sleeper_draft(draft_id, league_id, poll_interval)
        
    except Exception as e:
        click.echo(f"❌ Error connecting to Sleeper: {e}", err=True)

@draft.command()
@click.option('--draft-id', required=True, type=int, help='Draft ID')
@click.option('--team-position', type=int, help='Team position (defaults to current pick)')
@click.option('--limit', default=10, help='Number of recommendations to show')
@click.pass_context
def recommendations(ctx, draft_id, team_position, limit):
    """Get current draft recommendations"""
    try:
        click.echo(f"🤖 Getting recommendations for draft {draft_id}...")
        
        recs = ctx.obj['draft_engine'].get_recommendations(draft_id, team_position)
        
        if not recs:
            click.echo("❌ No recommendations available")
            click.echo("Make sure projections are aggregated and draft is active")
            return
        
        click.echo(f"\n💡 Top {min(limit, len(recs))} Recommendations:")
        click.echo(f"{'#':<2} {'Name':<25} {'Pos':<3} {'Value':<7} {'Reason':<35}")
        click.echo("─" * 80)
        
        for i, rec in enumerate(recs[:limit], 1):
            click.echo(f"{i:<2} {rec['name']:<25} {rec['position']:<3} "
                      f"{rec['value']:<7.1f} {rec['reason']:<35}")
        
        # Show draft context
        draft_info = ctx.obj['draft_engine'].get_draft_info(draft_id)
        if draft_info:
            current_pick = draft_info['current_pick']
            total_picks = draft_info['total_rounds'] * draft_info['num_teams']
            click.echo(f"\n📊 Draft Status: Pick {current_pick} of {total_picks}")
                      
    except Exception as e:
        click.echo(f"❌ Error getting recommendations: {e}", err=True)

@draft.command()
@click.pass_context
def list(ctx):
    """List all drafts"""
    try:
        query = """
        SELECT d.draft_id, d.league_id, d.status, d.current_pick, d.total_rounds,
               l.num_teams, l.scoring_system, d.created_at
        FROM drafts d
        JOIN leagues l ON d.league_id = l.league_id
        ORDER BY d.created_at DESC
        """
        
        drafts = ctx.obj['db'].execute_query(query)
        
        if not drafts:
            click.echo("No drafts found")
            return
        
        click.echo(f"\n📋 Draft History")
        click.echo(f"{'ID':<4} {'League':<15} {'Status':<10} {'Pick':<8} {'Teams':<6} {'Scoring':<8} {'Date':<12}")
        click.echo("─" * 75)
        
        for draft in drafts:
            created_date = draft['created_at'][:10] if draft['created_at'] else 'N/A'
            total_picks = draft['total_rounds'] * draft['num_teams']
            pick_status = f"{draft['current_pick']}/{total_picks}"
            
            click.echo(f"{draft['draft_id']:<4} {draft['league_id']:<15} "
                      f"{draft['status']:<10} {pick_status:<8} "
                      f"{draft['num_teams']:<6} {draft['scoring_system']:<8} {created_date:<12}")
                      
    except Exception as e:
        click.echo(f"❌ Error listing drafts: {e}", err=True)

@cli.command()
@click.option('--check-projections', is_flag=True, help='Check projection data')
@click.option('--check-database', is_flag=True, help='Check database health')
@click.pass_context
def status(ctx, check_projections, check_database):
    """Check system status"""
    try:
        click.echo("🔍 System Status Check\n")
        
        # Database connection
        try:
            ctx.obj['db'].execute_query("SELECT COUNT(*) as count FROM players")
            click.echo("✅ Database connection: OK")
        except Exception as e:
            click.echo(f"❌ Database connection: FAILED - {e}")
            return
        
        if check_database:
            # Check table counts
            tables = ['players', 'projections', 'projection_sources', 'aggregated_projections']
            click.echo("\n📊 Database Tables:")
            for table in tables:
                try:
                    result = ctx.obj['db'].execute_query(f"SELECT COUNT(*) as count FROM {table}")
                    count = result[0]['count'] if result else 0
                    click.echo(f"  {table:<20}: {count:>6} records")
                except Exception as e:
                    click.echo(f"  {table:<20}: ERROR - {e}")
        
        if check_projections:
            # Check projections status
            click.echo("\n🎯 Projections Status:")
            
            # Count by source
            source_query = """
            SELECT ps.name, COUNT(p.projection_id) as count
            FROM projection_sources ps
            LEFT JOIN projections p ON ps.source_id = p.source_id
            GROUP BY ps.name
            ORDER BY count DESC
            """
            
            sources = ctx.obj['db'].execute_query(source_query)
            for source in sources:
                click.echo(f"  {source['name']:<20}: {source['count']:>6} projections")
            
            # Check aggregated projections
            agg_result = ctx.obj['db'].execute_query("SELECT COUNT(*) as count FROM aggregated_projections")
            agg_count = agg_result[0]['count'] if agg_result else 0
            click.echo(f"  {'Aggregated':<20}: {agg_count:>6} projections")
            
            if agg_count == 0:
                click.echo("  ⚠️ No aggregated projections found. Run 'projections aggregate'")
        
        click.echo(f"\n✅ System check complete")
        
    except Exception as e:
        click.echo(f"❌ Error checking status: {e}", err=True)

if __name__ == '__main__':
    cli()