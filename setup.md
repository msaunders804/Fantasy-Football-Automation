# Fantasy Football Automation - Setup Instructions

## 🚀 Quick Setup Guide

### 1. Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)
- Web browser (Chrome, Firefox, Safari, Edge)
- Anthropic API account (for AI Advisor feature)

### 2. Installation Steps

#### Clone the Repository
```bash
git clone https://github.com/msaunders804/Fantasy-Football-Automation.git
cd Fantasy-Football-Automation
```

#### Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

#### Create .env File
```bash
# Copy the template
cp .env.template .env
```

#### Edit .env File
Open `.env` in your text editor and add:

```bash
# Required: Flask Secret Key
SECRET_KEY=your-super-secret-key-here-change-this

# Required: Anthropic API Key (for AI Advisor)
ANTHROPIC_API_KEY=sk-ant-api03-your-anthropic-key-here

# Optional: Development Settings
FLASK_ENV=development
DATABASE_URL=sqlite:///fantasy.db
```

### 4. Get Your Anthropic API Key

1. **Sign up at Anthropic Console**
   - Go to https://console.anthropic.com/
   - Create account and verify email
   - Add payment method (required even for free tier)

2. **Generate API Key**
   - Click "API Keys" in the dashboard
   - Click "Create Key"
   - Name it "fantasy-football-bot"
   - Copy the key (starts with `sk-ant-`)
   - Paste it in your `.env` file

3. **Free Credits**
   - New accounts get $5 in free credits
   - Enough for ~400-500 AI advice requests
   - Each chat interaction costs ~$0.01-0.03

### 5. Initialize Database
```bash
python app.py
# Database tables will be created automatically
# Stop the server with Ctrl+C after it starts successfully
```

### 6. Run the Application
```bash
python app.py
```

Open your browser and go to: http://localhost:5000

## 🎯 Features Overview

### ✅ Currently Available
- **AI Advisor**: Chat with Claude AI for fantasy advice
- **Draft Assistant UI**: Interactive draft setup (simulation & manual modes)
- **League Settings**: Configurable scoring and roster settings
- **Responsive Design**: Works on desktop, tablet, and mobile

### 🚧 Coming Soon (Integration Points)
- **ML Model Integration**: Connect your existing models
- **Player Database**: Connect your player data
- **Sleeper API**: Real-time league synchronization
- **Advanced Analytics**: Charts and visualizations

## 🔧 Configuration Options

### League Settings
Access via Settings modal (gear icon) or `/api/league-settings`:
- Number of teams (8, 10, 12, 14, 16)
- Scoring system (Standard, Half PPR, Full PPR)
- Roster requirements (QB, RB, WR, TE, FLEX, K, DEF)
- Current NFL week

### AI Advisor Settings
The AI advisor uses your league settings automatically for context:
- Roster composition analysis
- Position-specific advice
- League-aware recommendations

## 🚀 Development & Customization

### File Structure
```
Fantasy-Football-Automation/
├── app.py                 # Main Flask application
├── templates/            # HTML templates
│   ├── base.html         # Base template with navigation
│   ├── index.html        # Dashboard
│   ├── advice.html       # AI Advisor chat
│   ├── draft.html        # Draft assistant
│   ├── 404.html          # Error pages
│   └── 500.html
├── static/               # CSS, JS, images (create as needed)
├── fantasy.db            # SQLite database (auto-created)
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (create from template)
└── .gitignore           # Git ignore rules
```

### Integration Points
The following functions in `app.py` are ready for your data integration:

#### `get_fantasy_context(session_id)`
- Currently returns basic league settings and roster data
- **TODO**: Connect to your player projections and ML models

#### `get_user_roster()` & `update_roster()` 
- Basic roster management API endpoints
- **TODO**: Integrate with your existing player database

### Adding Your ML Models
1. Update `get_fantasy_context()` to include your model predictions
2. Add player projection endpoints in `app.py`
3. Connect to your existing draft logic

## 🐛 Troubleshooting

### Common Issues

#### "ModuleNotFoundError: No module named 'anthropic'"
```bash
# Make sure virtual environment is activated
pip install -r requirements.txt
```

#### "anthropic.APIError: API key not found"
- Check your `.env` file has `ANTHROPIC_API_KEY=sk-ant-...`
- Ensure the `.env` file is in the same directory as `app.py`
- Restart the Flask application after adding the key

#### "Chat not working"
- Verify your Anthropic API key is valid at https://console.anthropic.com/
- Check you have available credits
- Look at the browser console for JavaScript errors

#### Database Issues
```bash
# Delete and recreate database
rm fantasy.db
python app.py
# Database will be recreated automatically
```

### Performance Tips

#### Reduce API Costs
- Use shorter, more specific questions
- Clear chat history regularly (clears context)
- Monitor usage at https://console.anthropic.com/

#### Speed Optimization
- Keep league settings updated for better context
- Use quick action buttons instead of typing full questions

## 📱 Mobile Support

The application is fully responsive and works on:
- Desktop browsers (Chrome, Firefox, Safari, Edge)
- Tablet devices (iPad, Android tablets)
- Mobile phones (iOS Safari, Android Chrome)

## 🔒 Security Notes

- Never commit your `.env` file to Git
- Keep your API keys secure
- The app uses SQLite by default (single user)
- For production deployment, consider PostgreSQL

## 🆘 Support

If you encounter issues:

1. **Check the console logs** in your browser's developer tools
2. **Verify your API key** at the Anthropic console
3. **Review the Flask console output** for error messages
4. **Check your internet connection** for API calls

## 🎉 You're Ready!

Your Fantasy Football Automation app with AI Advisor is now running! 

- **Dashboard**: http://localhost:5000/
- **AI Advisor**: http://localhost:5000/advice
- **Draft Assistant**: http://localhost:5000/draft

Start by configuring your league settings, then chat with the AI advisor for personalized fantasy football advice!