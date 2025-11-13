"""
AutoGen Custom Demo - Conference Planning Workflow (Exercise 4)

This demonstrates a custom multi-agent workflow for planning a 3-day tech conference.
It shows how to adapt AutoGen for different problem domains beyond the original interview platform.

Agents:
1. ThemeAgent - Defines conference theme and target audience
2. SpeakerAgent - Identifies potential speakers and topics
3. ScheduleAgent - Creates detailed 3-day agenda
4. LogisticsAgent - Plans venue, catering, and logistics
5. MarketingAgent - Creates promotional strategy
"""

from datetime import datetime
from config import Config
import json

# Try to import OpenAI client
try:
    from openai import OpenAI
except ImportError:
    print("ERROR: OpenAI client is not installed!")
    print("Please run: pip install -r ../requirements.txt")
    exit(1)


class ConferencePlanningWorkflow:
    """Custom workflow for tech conference planning"""

    def __init__(self, conference_name: str, conference_type: str, duration: str = "3 days"):
        """Initialize the workflow"""
        if not Config.validate_setup():
            print("ERROR: Configuration validation failed!")
            exit(1)

        self.client = OpenAI(api_key=Config.API_KEY, base_url=Config.API_BASE)
        self.outputs = {}
        self.model = Config.OPENAI_MODEL
        self.conference_name = conference_name
        self.conference_type = conference_type
        self.duration = duration

    def run(self):
        """Execute the complete conference planning workflow"""
        print("\n" + "="*80)
        print(f"CONFERENCE PLANNING WORKFLOW - {self.conference_name.upper()}")
        print("="*80)
        print(f"Conference Type: {self.conference_type}")
        print(f"Duration: {self.duration}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model: {self.model}\n")

        # Phase 1: Theme & Vision
        self.phase_theme()

        # Phase 2: Speaker Curation
        self.phase_speakers()

        # Phase 3: Schedule Creation
        self.phase_schedule()

        # Phase 4: Logistics Planning
        self.phase_logistics()

        # Phase 5: Marketing Strategy
        self.phase_marketing()

        # Summary
        self.print_summary()

    def phase_theme(self):
        """Phase 1: Conference Theme & Vision"""
        print("\n" + "="*80)
        print("PHASE 1: CONFERENCE THEME & VISION")
        print("="*80)
        print("[ThemeAgent is defining the conference vision...]")

        system_prompt = f"""You are a conference theme strategist specializing in {self.conference_type} events.
Define a compelling conference theme and vision. Include:
- Conference theme/tagline (creative and memorable)
- Target audience (who should attend?)
- Key topics/tracks (3-4 main themes)
- Conference goals and value proposition
Keep it concise - 200 words."""

        user_message = f"Create a compelling theme and vision for '{self.conference_name}', a {self.duration} {self.conference_type} conference."

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=Config.AGENT_TEMPERATURE,
                max_tokens=Config.AGENT_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            )

            self.outputs["theme"] = response.choices[0].message.content
            print("\n[ThemeAgent Output]")
            print(self.outputs["theme"])
        except Exception as e:
            print(f"❌ Error: {e}")
            raise

    def phase_speakers(self):
        """Phase 2: Speaker & Topic Curation"""
        print("\n" + "="*80)
        print("PHASE 2: SPEAKER & TOPIC CURATION")
        print("="*80)
        print("[SpeakerAgent is identifying speakers and sessions...]")

        system_prompt = f"""You are a speaker curator for {self.conference_type} conferences.
Based on the conference theme, identify:
- 6-8 potential speaker profiles (roles/expertise, not specific names)
- Session types (keynotes, workshops, panels, lightning talks)
- Topic suggestions for each track
- Speaking format recommendations
Keep it concise - 200 words."""

        user_message = f"""Conference Theme and Vision:
{self.outputs['theme']}

Now identify ideal speaker profiles and session topics for this {self.duration} conference."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=Config.AGENT_TEMPERATURE,
                max_tokens=Config.AGENT_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            )

            self.outputs["speakers"] = response.choices[0].message.content
            print("\n[SpeakerAgent Output]")
            print(self.outputs["speakers"])
        except Exception as e:
            print(f"❌ Error: {e}")
            raise

    def phase_schedule(self):
        """Phase 3: Detailed Schedule Creation"""
        print("\n" + "="*80)
        print("PHASE 3: DETAILED SCHEDULE CREATION")
        print("="*80)
        print("[ScheduleAgent is creating the conference agenda...]")

        system_prompt = f"""You are a conference schedule planner.
Based on the theme and speaker information, create a detailed {self.duration} agenda:
- Day-by-day breakdown
- Time slots for each session (use realistic times: 9am-5pm)
- Balance of keynotes, workshops, panels, and networking
- Include breaks, meals, and networking time
- Note parallel tracks if applicable
Keep it structured and clear - 250 words."""

        user_message = f"""Conference Theme:
{self.outputs['theme']}

Speaker & Session Information:
{self.outputs['speakers']}

Create a detailed {self.duration} conference schedule."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=Config.AGENT_TEMPERATURE,
                max_tokens=Config.AGENT_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            )

            self.outputs["schedule"] = response.choices[0].message.content
            print("\n[ScheduleAgent Output]")
            print(self.outputs["schedule"])
        except Exception as e:
            print(f"❌ Error: {e}")
            raise

    def phase_logistics(self):
        """Phase 4: Logistics Planning"""
        print("\n" + "="*80)
        print("PHASE 4: LOGISTICS PLANNING")
        print("="*80)
        print("[LogisticsAgent is planning venue and logistics...]")

        system_prompt = """You are a conference logistics coordinator.
Based on the schedule and expected attendees, plan:
- Venue requirements (room sizes, AV equipment, layout)
- Catering needs (meals, breaks, dietary considerations)
- Registration and check-in process
- Technology requirements (WiFi, streaming, app)
- Contingency plans
Keep it practical and concise - 200 words."""

        user_message = f"""Conference Schedule:
{self.outputs['schedule']}

Conference Theme & Audience:
{self.outputs['theme']}

Plan the logistics for this conference."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=Config.AGENT_TEMPERATURE,
                max_tokens=Config.AGENT_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            )

            self.outputs["logistics"] = response.choices[0].message.content
            print("\n[LogisticsAgent Output]")
            print(self.outputs["logistics"])
        except Exception as e:
            print(f"❌ Error: {e}")
            raise

    def phase_marketing(self):
        """Phase 5: Marketing Strategy"""
        print("\n" + "="*80)
        print("PHASE 5: MARKETING STRATEGY")
        print("="*80)
        print("[MarketingAgent is creating promotional strategy...]")

        system_prompt = """You are a conference marketing strategist.
Based on all conference details, create a marketing plan:
- Key marketing messages and positioning
- Target promotion channels (social media, email, partnerships)
- Early bird vs regular pricing strategy
- Promotional timeline (3 months before event)
- Content marketing ideas
Keep it actionable - 200 words."""

        user_message = f"""Conference Details:

Theme & Vision:
{self.outputs['theme']}

Schedule Overview:
{self.outputs['schedule']}

Create a comprehensive marketing strategy for this conference."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=Config.AGENT_TEMPERATURE,
                max_tokens=Config.AGENT_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            )

            self.outputs["marketing"] = response.choices[0].message.content
            print("\n[MarketingAgent Output]")
            print(self.outputs["marketing"])
        except Exception as e:
            print(f"❌ Error: {e}")
            raise

    def print_summary(self):
        """Print final summary"""
        print("\n" + "="*80)
        print("CONFERENCE PLANNING SUMMARY")
        print("="*80)

        print(f"""
✅ Conference: {self.conference_name}
✅ Type: {self.conference_type}
✅ Duration: {self.duration}

This workflow demonstrated a 5-agent collaboration:
1. ThemeAgent - Defined conference vision and theme
2. SpeakerAgent - Curated speakers and topics
3. ScheduleAgent - Created detailed {self.duration} agenda
4. LogisticsAgent - Planned venue and logistics
5. MarketingAgent - Developed promotional strategy

Each agent built upon previous outputs to create a comprehensive conference plan.
""")

        # Print full results
        print("\n" + "="*80)
        print("FULL CONFERENCE PLAN - ALL COMPONENTS")
        print("="*80)
        
        sections = [
            ("THEME & VISION", "theme"),
            ("SPEAKERS & TOPICS", "speakers"),
            ("CONFERENCE SCHEDULE", "schedule"),
            ("LOGISTICS PLAN", "logistics"),
            ("MARKETING STRATEGY", "marketing")
        ]
        
        for title, key in sections:
            print("\n" + "-"*80)
            print(f"{title}")
            print("-"*80)
            print(self.outputs[key])

        # Save to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"conference_plan_{timestamp}.txt"
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"CONFERENCE PLANNING - {self.conference_name.upper()}\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.model}\n")
            f.write(f"Type: {self.conference_type}\n")
            f.write(f"Duration: {self.duration}\n\n")
            
            for title, key in sections:
                f.write("\n" + "-"*80 + "\n")
                f.write(f"{title}\n")
                f.write("-"*80 + "\n")
                f.write(self.outputs[key] + "\n")
        
        print(f"\n💾 Full conference plan saved to: {output_file}")
        print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)


if __name__ == "__main__":
    """
    Run custom conference planning scenarios.
    
    Try different conference types:
    - "AI/ML technology"
    - "web development"
    - "cybersecurity"
    - "product management"
    - "startup/entrepreneurship"
    """
    
    print("\n🎯 EXERCISE 4: CUSTOM PROBLEM - CONFERENCE PLANNING")
    print("="*80)
    print("This demo shows how to adapt AutoGen for different problem domains.")
    print("We're planning a multi-day tech conference using 5 specialized agents.")
    print("="*80)
    
    # Customize these parameters:
    CONFERENCE_NAME = "AI Horizons 2026"
    CONFERENCE_TYPE = "AI/ML technology"
    DURATION = "3 days"
    
    try:
        workflow = ConferencePlanningWorkflow(
            conference_name=CONFERENCE_NAME,
            conference_type=CONFERENCE_TYPE,
            duration=DURATION
        )
        workflow.run()
        print("\n✅ Conference planning workflow completed successfully!")
        print("\n💡 TIP: Edit the CONFERENCE_NAME, CONFERENCE_TYPE, and DURATION")
        print("    variables in this file to plan different types of conferences!")
    except Exception as e:
        print(f"\n❌ Error during workflow execution: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Verify your API key is set in .env")
        print("2. Check your API key has sufficient credits")
        print("3. Verify internet connection")
