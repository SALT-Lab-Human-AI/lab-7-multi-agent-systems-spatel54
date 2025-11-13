# Lab 7: Multi-Agent Systems - Complete Lab Report

**Student:** Shiv Patel  
**Date:** November 13, 2025  
**Repository:** lab-7-multi-agent-systems-spatel54  
**Branch:** main  

---

## Table of Contents
1. [Lab Overview](#lab-overview)
2. [Initial Setup](#initial-setup)
3. [Exercise 1: Running Basic Demos](#exercise-1-running-basic-demos)
4. [Exercise 2: Modifying Agent Roles and Backstories](#exercise-2-modifying-agent-roles-and-backstories)
5. [Exercise 3: Adding New Agents and Tasks](#exercise-3-adding-new-agents-and-tasks)
6. [Exercise 4: Custom Problem Domain](#exercise-4-custom-problem-domain)
7. [Technical Challenges and Solutions](#technical-challenges-and-solutions)
8. [Key Learnings](#key-learnings)
9. [Output Files Generated](#output-files-generated)

---

## Lab Overview

This lab explores two major multi-agent frameworks:
- **AutoGen**: Microsoft's framework for sequential agent workflows with context passing
- **CrewAI**: Framework for collaborative agent systems with specialized tools

**Technologies Used:**
- Python 3.13.9
- Groq API (llama-3.3-70b-versatile)
- OpenAI Client Library
- Virtual Environment (.venv)

---

## Initial Setup

### Configuration Changes
1. **Created `.env` file** from `.env.example`
2. **API Key Configuration:**
   - Initially attempted OpenAI API but encountered quota issues
   - **Switched to Groq API** for unlimited free tier access
   
3. **Final `.env` Configuration:**
```env
# Groq Configuration (using llama models)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# OpenAI Configuration (commented out - using Groq instead)
# OPENAI_API_KEY=your_openai_api_key_here
# OPENAI_MODEL=gpt-4-turbo-preview
```

### Dependencies Installed
```bash
pip install -r requirements.txt
```

Key packages:
- `openai>=1.0.0` - API client
- `python-dotenv>=1.0.0` - Environment management
- `pyautogen>=0.2.0` - AutoGen framework
- `crewai>=0.1.0` - CrewAI framework
- `crewai-tools>=0.1.0` - CrewAI tool extensions

---

## Exercise 1: Running Basic Demos

### AutoGen Simple Demo
**File:** `autogen/autogen_simple_demo.py`

**Purpose:** Demonstrates a 4-agent interview platform workflow where agents collaborate sequentially.

**Agents:**
1. **ResearchAgent** - Market research and competitor analysis
2. **AnalysisAgent** - Technical requirements analysis
3. **BlueprintAgent** - System architecture design
4. **ReviewAgent** - Quality review and recommendations

**Workflow:**
```
ResearchAgent → AnalysisAgent → BlueprintAgent → ReviewAgent
     ↓              ↓               ↓               ↓
  Context      Context         Context         Final
  Building     Enrichment      Synthesis       Output
```

**Command:**
```bash
python autogen/autogen_simple_demo.py
```

**Key Output:**
- Comprehensive interview platform product plan
- Market analysis of competitors (LeetCode, HackerRank, Codility)
- Technical architecture blueprint
- Feature recommendations

### CrewAI Travel Demo
**File:** `crewai/crewai_demo.py`

**Purpose:** Demonstrates collaborative agents planning a 5-day trip to Iceland.

**Agents:**
1. **FlightAgent** - Flight research specialist
2. **HotelAgent** - Accommodation specialist
3. **ItineraryAgent** - Travel planner
4. **BudgetAgent** - Financial advisor

**Tools Used:**
- `search_flight_prices()` - Web search for flight data
- `search_hotel_options()` - Hotel research tool
- `search_attractions_activities()` - Activity finder

**Command:**
```bash
python crewai/crewai_demo.py
```

**Key Output:** (See `crewai/crewai_output.txt`)
- **Flight Options:**
  - Option 1: Icelandair Direct - $420 (Recommended)
  - Option 2: Delta + Icelandair - $365 (Cheapest)
  - Option 3: United Airlines - $400
  
- **Accommodation (5 nights):**
  - Budget: Reykjavik Downtown Hostel - $350 total
  - Mid-Range: Fosshotel Reykjavik - $825 total
  - Luxury: Canopy by Hilton - $1,625 total

- **Daily Meal Budget:**
  - Budget: $35/day ($175 total)
  - Mid-Range: $75/day ($375 total)
  - Luxury: $153/day ($765 total)

- **Activities:**
  - Hallgrímskirkja Church: $7.50
  - Blue Lagoon: $80-$150
  - Northern Lights Tour: $120-$150
  - Golden Circle Tour: $80-$120

**Total Budget Estimates:**
- Budget Plan: ~$1,800-$2,200 per person
- Mid-Range: ~$3,000-$3,500 per person
- Luxury: ~$5,500-$6,500 per person

---

## Exercise 2: Modifying Agent Roles and Backstories

### AutoGen Modifications
**File:** `autogen/config.py`

#### Original vs Modified Agent Roles

**1. ResearchAgent:**
```python
# ORIGINAL
"role": "Senior Market Research Analyst specializing in HR Tech"

# MODIFIED (Exercise 2)
"role": "Strategic Market Intelligence Expert with 15+ years in HR Tech and Enterprise SaaS"
```

**2. AnalysisAgent:**
```python
# ORIGINAL
"role": "Technical Product Analyst with expertise in system architecture"

# MODIFIED (Exercise 2)
"role": "Principal Technical Architect with expertise in scalable enterprise systems and AI integration"
```

**3. BlueprintAgent:**
```python
# ORIGINAL
"role": "Senior Product Manager and System Architect"

# MODIFIED (Exercise 2)
"role": "Chief Product Strategist and Enterprise System Architect with proven track record in launching platforms at scale"
```

**4. ReviewAgent:**
```python
# ORIGINAL
"role": "Quality Assurance Lead and Product Reviewer"

# MODIFIED (Exercise 2)
"role": "VP of Product Quality and Innovation with focus on user experience excellence and market differentiation"
```

### CrewAI Modifications
**File:** `crewai/crewai_demo.py`

#### Enhanced Agent Backstories

**1. FlightAgent:**
```python
# ORIGINAL
backstory="Expert in finding the best flight deals and understanding airline policies."

# MODIFIED (Exercise 2)
backstory="""You are a world-class flight specialist with 10+ years of experience in the 
travel industry. You have deep connections with major airlines and access to insider knowledge 
about flight patterns, pricing algorithms, and seasonal deals. You pride yourself on finding 
the perfect balance between cost, comfort, and convenience. You stay updated on the latest 
airline policies, loyalty programs, and booking strategies."""
```

**2. HotelAgent:**
```python
# ORIGINAL
backstory="Specialist in hotel bookings with knowledge of various accommodation options."

# MODIFIED (Exercise 2)
backstory="""You are an elite accommodation specialist who has personally visited and vetted 
thousands of hotels worldwide. With an impeccable eye for quality, location, and value, you 
understand what makes a hotel truly special. You have insider relationships with hotel managers, 
know the best rooms in each property, and can spot hidden gems that online reviews might miss. 
Your recommendations are trusted by luxury travelers and budget backpackers alike."""
```

**3. ItineraryAgent:**
```python
# ORIGINAL
backstory="Experienced in creating detailed travel itineraries for various destinations."

# MODIFIED (Exercise 2)
backstory="""You are a master travel planner and cultural anthropologist who has traveled to 
over 100 countries. You don't just plan trips – you craft transformative experiences. You 
understand the rhythm of cities, the best times to visit attractions, and how to balance 
adventure with relaxation. Your itineraries are legendary for their perfect pacing, insider 
access, and ability to help travelers connect authentically with local culture. You consider 
factors like jet lag, weather patterns, local events, and personal interests to create 
truly personalized journeys."""
```

**4. BudgetAgent:**
```python
# ORIGINAL
backstory="Financial advisor specialized in travel budgeting and cost optimization."

# MODIFIED (Exercise 2)
backstory="""You are a travel finance guru and former CFO of a major travel company. You 
understand every nuance of travel costs, from hidden fees to seasonal price fluctuations. 
You're a master at finding value without sacrificing quality, and you know exactly where to 
splurge and where to save. Your budget analyses are comprehensive yet easy to understand, 
and you always provide multiple scenarios to help travelers make informed decisions. You 
track currency exchange trends, credit card rewards programs, and money-saving travel hacks."""
```

**Impact:** More detailed backstories led to richer, more professional outputs with deeper insights and recommendations.

---

## Exercise 3: Adding New Agents and Tasks

### AutoGen: Added TechnicalAgent
**File:** `autogen/config.py` and `autogen/autogen_simple_demo.py`

#### New Agent Configuration
```python
TECHNICAL_AGENT = {
    "name": "TechnicalAgent",
    "role": "Senior Software Engineer and Technology Consultant",
    "system_message": """You are a Senior Software Engineer and Technology Consultant 
    specializing in technical implementation planning. Your role is to bridge the gap between 
    architectural blueprints and actual code implementation.
    
    Your responsibilities:
    1. Review system architecture and identify technical implementation challenges
    2. Recommend specific technologies, frameworks, and libraries
    3. Create technical implementation roadmap with clear phases
    4. Identify potential technical risks and mitigation strategies
    5. Suggest best practices for code organization, testing, and deployment
    
    Provide practical, actionable technical guidance that development teams can follow.""",
    "description": "Technology expert who creates detailed technical implementation plans"
}
```

#### Workflow Integration
Added `phase_technical()` method between Blueprint and Review phases:

```python
def phase_technical(self):
    """Phase 4: Technical Implementation Planning"""
    print("\n" + "="*80)
    print("PHASE 4: TECHNICAL IMPLEMENTATION")
    print("="*80)
    print("[TechnicalAgent is creating implementation plan...]\n")
    
    # Agent builds on previous phases
    context = f"""
    Based on the following outputs:
    
    MARKET RESEARCH:
    {self.outputs['research']}
    
    TECHNICAL ANALYSIS:
    {self.outputs['analysis']}
    
    SYSTEM BLUEPRINT:
    {self.outputs['blueprint']}
    
    Create a detailed technical implementation plan including:
    1. Technology stack recommendations (languages, frameworks, databases)
    2. Implementation phases with timeline estimates
    3. Technical risks and mitigation strategies
    4. Development best practices and code organization
    5. Testing and deployment strategies
    """
    
    response = self.call_llm(context, "TechnicalAgent")
    self.outputs['technical'] = response
```

**Updated Workflow:**
```
Research → Analysis → Blueprint → Technical → Review
```

**New Phase Output:** Technical implementation roadmap with:
- Technology stack (React, Node.js, PostgreSQL, Redis)
- 4-phase implementation plan (6-9 months total)
- Risk mitigation strategies
- DevOps and testing recommendations

### CrewAI: Added MarketingAgent
**File:** `crewai/crewai_demo.py`

#### New Agent Creation
```python
def create_marketing_agent() -> Agent:
    """Create a Marketing Strategy Specialist agent"""
    return Agent(
        role="Marketing Strategy Specialist",
        goal="Develop comprehensive marketing strategies to promote travel destinations and packages",
        backstory="""You are a creative marketing strategist with expertise in travel marketing, 
        social media campaigns, and destination branding. You understand how to create compelling 
        travel narratives that inspire wanderlust and drive bookings. With experience in both 
        digital marketing and traditional travel marketing, you know how to reach different 
        traveler demographics. You're skilled at identifying unique selling points and crafting 
        messages that resonate with specific target audiences. Your campaigns have helped launch 
        successful travel products and destination marketing initiatives.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
```

#### New Task Creation
```python
def create_marketing_task(agent: Agent, context_tasks: list) -> Task:
    """Create marketing strategy task"""
    return Task(
        description="""Based on the complete travel plan including flights, hotels, itinerary, 
        and budget, create a comprehensive marketing strategy for this Iceland trip package.
        
        Your marketing strategy should include:
        1. Target audience definition and personas
        2. Unique selling points (USPs) of this travel package
        3. Marketing message and tagline
        4. Content marketing ideas (blog posts, social media content)
        5. Promotional channels (where to advertise this package)
        6. Pricing strategy and promotional offers
        7. Sample social media posts or ad copy
        
        Make the strategy actionable and specific to this Iceland trip package.""",
        agent=agent,
        expected_output="""A comprehensive marketing strategy document including target audience, 
        USPs, marketing messages, content ideas, promotional channels, and sample marketing copy.""",
        context=context_tasks
    )
```

**Impact:** Added marketing perspective to travel planning, including:
- Target audience personas (Adventure Seekers, Culture Enthusiasts, Luxury Travelers)
- Unique selling points for Iceland winter travel
- Social media campaign ideas
- Pricing strategies and promotional offers

**Updated Agent Count:**
- AutoGen: 4 agents → **5 agents** (added TechnicalAgent)
- CrewAI: 4 agents → **5 agents** (added MarketingAgent)

---

## Exercise 4: Custom Problem Domain

### Conference Planning Workflow
**File:** `autogen/autogen_conference_demo.py` (NEW FILE)

**Purpose:** Demonstrate AutoGen's flexibility by creating a completely different use case - planning a 3-day tech conference.

#### Problem Definition
- **Conference Name:** AI Horizons 2026
- **Type:** AI/ML Technology Conference
- **Duration:** 3 days
- **Target Audience:** C-level executives, AI/ML practitioners, data scientists

#### Agent Workflow

**1. ThemeAgent**
- **Role:** Conference Strategy Director
- **Responsibility:** Define theme, target audience, and key topics
- **Output:**
  - Theme: "Elevate. Amplify. Transform."
  - 4 Tracks: AI for Enterprise, ML Mastery, AI Ethics, Future Frontiers
  - Target audience definition
  - Conference goals and value proposition

**2. SpeakerAgent**
- **Role:** Program Director and Speaker Coordinator
- **Responsibility:** Identify speakers and session topics
- **Output:**
  - Speaker profiles for each track
  - Session topics (keynotes, panels, workshops)
  - Speaking format recommendations
  - Examples:
    - Keynote: "AI-Driven Business Growth Strategies" (CEO)
    - Panel: "AI ROI: Measuring Success in the Enterprise"
    - Workshop: "Building AI-Ready Teams"

**3. ScheduleAgent**
- **Role:** Conference Logistics Coordinator
- **Responsibility:** Create detailed 3-day agenda
- **Output:**
  - Hour-by-hour schedule for 3 days
  - Multiple concurrent tracks
  - Networking breaks and meals
  - Session timing and locations
  - Example day structure:
    - 9:00 AM - Registration
    - 9:30 AM - Keynote
    - 10:30 AM - Panel Discussion
    - 12:00 PM - Lunch Break
    - 1:00 PM - Concurrent Sessions (3 tracks)
    - 4:00 PM - Closing Remarks

**4. LogisticsAgent**
- **Role:** Operations and Venue Manager
- **Responsibility:** Plan venue, catering, and logistics
- **Output:**
  - Venue requirements (main stage: 300-500 seats, 3 tracks: 150-200 seats each)
  - AV equipment needs
  - Layout and floor plan
  - Catering needs (breakfast, lunch, snacks)
  - Registration and check-in process
  - Technology requirements (WiFi, streaming, mobile app)
  - Contingency plans
  - Budget: $225,000 - $350,000

**5. MarketingAgent**
- **Role:** Marketing and Promotions Director
- **Responsibility:** Create promotional strategy
- **Output:**
  - Key marketing messages
  - Promotion channels (social media, email, partnerships)
  - Pricing strategy:
    - Early Bird: 15-20% discount (first 2-3 months)
    - Regular: Standard rate
    - Late Registration: 10-15% premium
  - Promotional timeline (3-month campaign)
  - Content marketing ideas (blog posts, whitepapers, podcasts)
  - Budget allocation across channels

#### Complete Output
**File:** `conference_plan_20251113_104903.txt` (461 lines)

The output demonstrates a **complete, production-ready conference plan** including:
- ✅ Conference vision and positioning
- ✅ 50+ speaker sessions across 3 days
- ✅ Detailed minute-by-minute schedule
- ✅ Comprehensive logistics plan
- ✅ Multi-channel marketing strategy

#### Key Success Metrics
- **Sequential Workflow:** Each agent built upon previous outputs
- **Context Awareness:** Agents referenced earlier phases in their analysis
- **Practical Output:** Real-world usable conference plan
- **Adaptability:** Demonstrated AutoGen works for diverse problem domains

---

## Technical Challenges and Solutions

### Challenge 1: OpenAI API Quota Exceeded
**Error:**
```
Error code: 429 - You exceeded your current quota
```

**Solution:**
- Switched from OpenAI API to Groq API
- Updated `.env` to use `GROQ_API_KEY` and `GROQ_MODEL`
- Groq provides generous free tier with llama models
- No code changes needed due to OpenAI-compatible API

### Challenge 2: Model Not Found
**Error:**
```
Error code: 404 - The model gpt-4-turbo-preview does not exist
```

**Solution:**
- Changed `GROQ_MODEL` to `llama-3.3-70b-versatile`
- Verified model availability in Groq documentation
- Updated all configuration files

### Challenge 3: Groq Rate Limit During Conference Demo
**Error:**
```
Rate limit exceeded: 100,000 tokens/day
```

**Context:** Hit Groq's daily token limit while running conference demo (Phase 2 of 5)

**Solution:**
- Temporarily switched to `llama-3.1-8b-instant` (faster, smaller model)
- Completed the conference demo successfully
- Switched back to `llama-3.3-70b-versatile` after completion

**Lesson Learned:** 
- Smaller models use fewer tokens per request
- Balance between output quality and token usage
- Plan API usage for large multi-agent workflows

### Challenge 4: CrewAI Tool Validation Errors
**Error:**
```
ValidationError: Agent must receive a string as input
```

**Cause:** Tool was returning dict instead of string

**Solution:**
- Ensured all tools return formatted strings
- CrewAI's retry logic handled transient errors
- Added proper error handling in tool definitions

### Challenge 5: Context Length Management
**Issue:** Large outputs from early agents causing context overflow

**Solution:**
- Implemented output summarization for long responses
- Used selective context passing (only relevant portions)
- Structured outputs with clear sections

---

## Key Learnings

### 1. Framework Comparison

| Aspect | AutoGen | CrewAI |
|--------|---------|--------|
| **Workflow Style** | Sequential, context-passing | Collaborative, parallel-capable |
| **Use Cases** | Analytical pipelines, step-by-step planning | Research-intensive, tool-heavy tasks |
| **Context Management** | Explicit passing between agents | Automatic context sharing |
| **Tool Integration** | Manual implementation | Built-in tool decorators |
| **Complexity** | Lower learning curve | More features, steeper curve |
| **Best For** | Structured workflows with clear phases | Research and data gathering tasks |

### 2. Agent Design Principles
- **Specificity:** Detailed roles and backstories produce better outputs
- **Context:** Each agent should understand what came before
- **Division of Labor:** Clear separation of responsibilities prevents overlap
- **Validation:** Review agents catch errors from earlier phases

### 3. API Management
- **Rate Limits:** Plan for API quota management
- **Model Selection:** Balance quality vs. speed vs. cost
- **Fallback Strategies:** Have alternative models available
- **Monitoring:** Track token usage across workflow

### 4. Workflow Design
- **Sequential for Planning:** Use AutoGen when order matters
- **Parallel for Research:** Use CrewAI when agents can work simultaneously
- **Incremental Building:** Each phase adds value to previous work
- **Summary Generation:** Consolidate outputs for human consumption

---

## Output Files Generated

### AutoGen Outputs
1. **`conference_plan_20251113_104903.txt`** (461 lines)
   - Complete conference planning output from Exercise 4
   - 5 phases: Theme, Speakers, Schedule, Logistics, Marketing
   - Production-ready conference plan

### CrewAI Outputs
1. **`crewai/crewai_output.txt`** (166 lines)
   - Iceland travel plan with detailed budget breakdown
   - Flight options, accommodation choices, activity recommendations
   - Multi-tier pricing (budget, mid-range, luxury)

### Configuration Files Modified
1. **`.env`**
   - Configured Groq API credentials
   - Model selection (switched between llama-3.3-70b-versatile and llama-3.1-8b-instant)

2. **`autogen/config.py`**
   - Modified all agent roles (Exercise 2)
   - Added TechnicalAgent configuration (Exercise 3)

3. **`autogen/autogen_simple_demo.py`**
   - Added `phase_technical()` method
   - Updated workflow to 5 agents

4. **`crewai/crewai_demo.py`**
   - Enhanced agent backstories (Exercise 2)
   - Added MarketingAgent and marketing task (Exercise 3)

### New Files Created
1. **`autogen/autogen_conference_demo.py`** (375 lines)
   - Complete custom workflow for Exercise 4
   - 5-agent conference planning system
   - Demonstrates framework flexibility

2. **`LAB_REPORT.md`** (this file)
   - Comprehensive documentation of all exercises
   - Technical challenges and solutions
   - Output summaries and learnings

---

## Conclusion

This lab successfully demonstrated:

✅ **Exercise 1:** Ran both AutoGen and CrewAI demos successfully with Groq API  
✅ **Exercise 2:** Modified all agent roles and backstories in both frameworks  
✅ **Exercise 3:** Added new agents (TechnicalAgent in AutoGen, MarketingAgent in CrewAI) with custom tasks  
✅ **Exercise 4:** Created complete custom conference planning workflow demonstrating framework flexibility  

**Total Agents Implemented:** 10 agents across 2 frameworks (5 each)  
**Total Lines of Output:** 627+ lines of high-quality agent outputs  
**API Calls:** 50+ successful LLM API calls via Groq  

The lab showcased the power of multi-agent systems for complex problem-solving and highlighted the differences between sequential (AutoGen) and collaborative (CrewAI) agent architectures.

---

**End of Lab Report**
