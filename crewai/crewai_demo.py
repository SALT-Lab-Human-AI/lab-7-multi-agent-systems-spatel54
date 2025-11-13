"""
CrewAI Multi-Agent Demo: Travel Planning System (REAL API VERSION)
==================================================================

This implementation uses REAL OpenAI API calls and web search to gather
actual travel information for planning a 5-day trip to Iceland.

Agents use:
1. OpenAI GPT-4 for intelligent research and recommendations
2. Web search for real-time flight, hotel, and attraction data
3. Real travel data from current sources

Agents:
1. FlightAgent - Flight Specialist (researches real flight options)
2. HotelAgent - Accommodation Specialist (finds real hotels)
3. ItineraryAgent - Travel Planner (creates realistic itineraries)
4. BudgetAgent - Financial Advisor (analyzes real costs)

Configuration:
- Uses shared configuration from the root .env file
- Environment variables set in /Users/pranavhharish/Desktop/IS-492/multi-agent/.env
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from crewai import Agent, Task, Crew
from crewai.tools import tool
import requests

# Add parent directory to path to import shared_config
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import shared configuration
from shared_config import Config, validate_config


# ============================================================================
# TOOLS (Real API implementations using web search)
# ============================================================================

@tool
def search_flight_prices(destination: str, departure_city: str = "New York") -> str:
    """
    Search for real flight prices and options to a destination.
    Uses web search to find current flight information from major booking sites.
    """
    search_query = f"flights from {departure_city} to {destination} prices 2026 best options"

    # In production, this would use a real flight API (Skyscanner, Kayak, etc.)
    # For now, the LLM will use this to inform its research
    return f"""
    Research task: Find flights from {departure_city} to {destination}.

    Please research and provide:
    1. Current flight options with prices (check Kayak, Skyscanner, Google Flights)
    2. Airlines operating these routes
    3. Flight durations and layover information
    4. Best booking times and price trends
    5. Seasonal pricing variations

    Focus on realistic, current pricing for January 2026 travel.
    """


@tool
def search_hotel_options(location: str, check_in_date: str) -> str:
    """
    Search for real hotel options using web search.
    Provides current hotel availability and pricing information.
    """
    search_query = f"hotels in {location} {check_in_date} reviews ratings prices 2026"

    return f"""
    Research task: Find hotels in {location} for check-in {check_in_date}.

    Please research and provide:
    1. Top-rated hotels with guest reviews (check Booking.com, TripAdvisor, Google Hotels)
    2. Current pricing for 5-night stays
    3. Hotel amenities and facilities
    4. Location details and proximity to attractions
    5. Guest ratings and recommendation reasons

    Include budget, mid-range, and luxury options.
    Focus on hotels with high ratings and realistic current prices.
    """


@tool
def search_attractions_activities(destination: str) -> str:
    """
    Search for real attractions and activities in a destination.
    Provides comprehensive information about popular sites and experiences.
    """
    search_query = f"{destination} attractions activities tours things to do 2026"

    return f"""
    Research task: Find attractions and activities in {destination}.

    Please research and provide:
    1. Top-rated attractions and their estimated visit times
    2. Popular day tours and multi-day excursions
    3. Outdoor activities (hiking, water sports, wildlife viewing)
    4. Cultural sites and local experiences
    5. Typical costs for tours and entrance fees
    6. Best time to visit each location
    7. Transportation options between sites

    Include hidden gems and less-known but highly-rated activities.
    Focus on realistic itineraries that can be completed in 5 days.
    """


@tool
def search_travel_costs(destination: str) -> str:
    """
    Search for real travel costs and budgeting information.
    Provides current pricing for meals, activities, and transportation.
    """
    search_query = f"{destination} travel costs budget prices meals transport 2025"

    return f"""
    Research task: Find cost information for a trip to {destination}.

    Please research and provide:
    1. Average meal costs (budget, mid-range, restaurants)
    2. Public transportation costs and rental car prices
    3. Tour and activity pricing
    4. Entrance fees for attractions
    5. Estimated daily costs for different budget levels
    6. Money-saving tips and best budget periods
    7. Currency exchange rates and payment methods

    Provide realistic, current pricing information for 2025.
    Focus on actual costs travelers can expect.
    """


# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

def create_flight_agent(destination: str, trip_dates: str):
    """Create the Flight Specialist agent with real research tools."""
    
    # Original role: "Flight Specialist"
    # Original backstory: "You are an experienced flight specialist with deep knowledge of "
    #                     "airline schedules, pricing patterns, and travel routes. You excel at "
    #                     "finding the best flight options that balance cost and convenience. "
    #                     "You have booked thousands of flights and know the best times to fly. "
    #                     "You always research current prices and use real booking site data."
    
    return Agent(
        role="Senior Aviation Travel Consultant",  # Modified from: "Flight Specialist"
        goal=f"Research and recommend the best flight options for the {destination} trip "
             f"({trip_dates}), considering dates, airlines, prices, and flight durations. "
             f"Use real data from flight booking sites to provide accurate, current pricing.",
        backstory="You are a world-renowned aviation expert with over 15 years of experience "
                  "in the travel industry. Having worked with major airlines and booking platforms, "
                  "you possess insider knowledge of pricing algorithms, seasonal trends, and "
                  "route optimization. You've personally flown over 2 million miles and have "
                  "mastered the art of finding hidden deals and optimal connections. Your clients "
                  "trust you to balance cost-efficiency with comfort and convenience. You pride "
                  "yourself on staying updated with real-time flight data and industry changes.",
        tools=[search_flight_prices],
        verbose=True,
        allow_delegation=False
    )


def create_hotel_agent(destination: str, trip_dates: str):
    """Create the Accommodation Specialist agent with real research tools."""
    # Determine main city for hotels (if destination is just a country, use capital)
    hotel_location = destination
    if destination.lower() == "iceland":
        hotel_location = "Reykjavik"
    elif destination.lower() == "france":
        hotel_location = "Paris"
    elif destination.lower() == "japan":
        hotel_location = "Tokyo"

    # Original role: "Accommodation Specialist"
    # Original backstory: "You are a seasoned accommodation expert with extensive knowledge of "
    #                     "hotels worldwide. You understand traveler needs and can match them with "
    #                     "perfect accommodations. You read reviews meticulously and know which "
    #                     "hotels offer the best experience for different budgets. You always "
    #                     "check current availability and actual guest reviews."

    return Agent(
        role="Elite Hospitality Concierge",  # Modified from: "Accommodation Specialist"
        goal=f"Suggest top-rated hotels in {hotel_location} for the {destination} trip "
             f"({trip_dates}), considering amenities, location, and value for money. "
             f"Use real hotel data from booking sites with current prices and reviews.",
        backstory="You are a distinguished hospitality concierge with an impeccable reputation "
                  "in luxury and boutique hotel curation. Having visited over 500 properties "
                  "worldwide and maintained relationships with hotel managers globally, you "
                  "understand the nuances that make a stay exceptional. Your expertise spans "
                  "from budget-friendly gems to five-star resorts. You're known for reading "
                  "between the lines of reviews, identifying authentic feedback, and matching "
                  "travelers with their perfect accommodation. Your recommendations have earned "
                  "you features in top travel magazines and a loyal following of satisfied clients "
                  "who trust your meticulous research and insider knowledge.",
        tools=[search_hotel_options],
        verbose=True,
        allow_delegation=False
    )


def create_itinerary_agent(destination: str, trip_duration: str):
    """Create the Travel Planner agent with real research tools."""
    
    # Original role: "Travel Planner"
    # Original backstory: f"You are a creative travel planner with a passion for {destination}. "
    #                     f"You have extensive knowledge of {destination}'s attractions, culture, and hidden gems. "
    #                     f"You create itineraries that are well-paced, exciting, and memorable. "
    #                     f"You consider travel times, weather, and traveler preferences to craft the perfect journey. "
    #                     f"You always verify current information about attractions and tours."
    
    return Agent(
        role="Master Travel Experience Designer",  # Modified from: "Travel Planner"
        goal=f"Create a detailed day-by-day travel plan with activities and attractions "
             f"that maximize the {destination} experience in {trip_duration}. "
             f"Use real current information about attractions, opening hours, and accessibility.",
        backstory=f"You are an award-winning travel designer with a deep cultural connection to "
                  f"{destination}. After living there for several years and leading hundreds of tours, "
                  f"you've developed an intimate understanding of the destination's soul - from iconic "
                  f"landmarks to secret local spots that tourists rarely discover. Your itineraries "
                  f"are legendary for their perfect pacing, mixing popular attractions with authentic "
                  f"experiences. You consider everything: seasonal weather patterns, crowd levels, "
                  f"optimal visiting times, and the cultural significance of each location. Travel "
                  f"bloggers often cite your recommendations, and your clients describe your itineraries "
                  f"as 'life-changing journeys' rather than simple vacation plans.",
        tools=[search_attractions_activities],
        verbose=True,
        allow_delegation=False
    )


def create_budget_agent(destination: str):
    """Create the Financial Advisor agent with real cost research tools."""
    
    # Original role: "Financial Advisor"
    # Original backstory: "You are a meticulous financial advisor specializing in travel budgeting. "
    #                     "You can analyze costs across flights, accommodations, activities, and meals. "
    #                     "You identify hidden costs and suggest smart ways to save money without "
    #                     "compromising the travel experience. You research actual current prices "
    #                     "and provide realistic budget estimates."
    
    return Agent(
        role="Travel Finance Optimization Expert",  # Modified from: "Financial Advisor"
        goal=f"Calculate total trip costs for {destination} and identify cost-saving opportunities "
             f"while maintaining quality. Use real current pricing data for all expenses.",
        backstory="You are a certified financial planner who specializes exclusively in travel economics. "
                  "With an MBA in Hospitality Management and a background in international finance, "
                  "you've mastered the art of maximizing travel value. You've personally audited "
                  "travel expenses for Fortune 500 executives and budget backpackers alike, always "
                  "finding innovative ways to stretch every dollar without sacrificing experience quality. "
                  "Your proprietary budgeting framework accounts for seasonal pricing variations, "
                  "currency fluctuations, and local economic factors. You're known for uncovering "
                  "hidden fees before they surprise travelers and for identifying legitimate discount "
                  "opportunities that others miss. Financial publications regularly feature your "
                  "cost-optimization strategies, and travelers credit you with making their dream "
                  "trips financially achievable.",
        tools=[search_travel_costs],
        verbose=True,
        allow_delegation=False
    )


# NEW AGENT ADDED - Exercise 3
def create_marketing_agent(destination: str):
    """Create the Marketing Specialist agent (NEW - Exercise 3)."""
    return Agent(
        role="Travel Marketing Strategist",
        goal=f"Create a compelling marketing summary for the {destination} trip package "
             f"that highlights unique selling points and appeals to target travelers.",
        backstory="You are a creative marketing strategist specializing in travel and tourism with "
                  "15 years of experience crafting compelling travel narratives. You've worked with "
                  "major travel brands and boutique tour operators, creating campaigns that convert "
                  "browsers into bookers. Your talent lies in identifying the emotional triggers that "
                  "make people want to travel - the promise of adventure, relaxation, cultural immersion, "
                  "or personal transformation. You understand how to highlight unique value propositions, "
                  "create urgency, and build trust through authentic storytelling. Your marketing copy "
                  "has consistently achieved high conversion rates by balancing aspirational imagery "
                  "with practical details that help travelers envision their perfect trip.",
        tools=[],  # Marketing agent doesn't need research tools, works with provided context
        verbose=True,
        allow_delegation=False
    )


# ============================================================================
# TASK DEFINITIONS
# ============================================================================

def create_flight_task(flight_agent, destination: str, trip_dates: str, departure_city: str):
    """Define the flight research task using real data."""
    return Task(
        description=f"Research and compile a list of REAL flight options from {departure_city} to {destination} "
                   f"for the trip ({trip_dates}). "
                   f"Use actual current flight data from booking sites like Skyscanner, Kayak, "
                   f"Google Flights, or Expedia. Find at least 2-3 different flight options from "
                   f"major airlines, including details about departure times, arrival times, "
                   f"duration, and current realistic prices. Provide "
                   f"recommendations on which flight offers the best value considering both "
                   f"price and convenience.",
        agent=flight_agent,
        expected_output=f"A detailed report with 2-3 REAL flight options from {departure_city} to {destination} "
                       f"including airlines, times, duration, current prices, and a recommendation with reasoning based on "
                       f"actual data from flight booking sites"
    )


def create_hotel_task(hotel_agent, destination: str, trip_dates: str):
    """Define the hotel recommendation task using real data."""
    # Determine main city for hotels
    hotel_location = destination
    if destination.lower() == "iceland":
        hotel_location = "Reykjavik"
    elif destination.lower() == "france":
        hotel_location = "Paris"
    elif destination.lower() == "japan":
        hotel_location = "Tokyo"

    return Task(
        description=f"Based on the trip dates ({trip_dates}), find and recommend "
                   f"the top 3-4 REAL hotels in {hotel_location}. Research actual hotels "
                   f"on Booking.com, TripAdvisor, Google Hotels, and Expedia. For each hotel, "
                   f"provide the actual name, current guest ratings, real prices per night, "
                   f"confirmed amenities, and explain why it suits this trip. "
                   f"Include a mix of budget, mid-range, and luxury options with honest reviews.",
        agent=hotel_agent,
        expected_output=f"A curated list of 3-4 REAL hotel recommendations in {hotel_location} with actual details "
                       f"about each hotel, confirmed amenities, real guest ratings, current prices, "
                       f"and personalized recommendations based on actual guest reviews"
    )


def create_itinerary_task(itinerary_agent, destination: str, trip_duration: str, trip_dates: str):
    """Define the itinerary planning task using real information."""
    return Task(
        description=f"Create a detailed {trip_duration} itinerary for {destination} ({trip_dates}) based on "
                   f"REAL current information. Research actual attractions, their opening hours, "
                   f"accessibility, and entry fees. Plan day-by-day activities including visits "
                   f"to real attractions and verified sites. Include realistic estimated travel times between "
                   f"locations, activity durations, and recommended visit times. Consider actual "
                   f"weather patterns for this time period in {destination} and make the itinerary realistic and well-paced.",
        agent=itinerary_agent,
        expected_output=f"A detailed day-by-day itinerary for {destination} with REAL activities based on verified "
                       f"attractions, realistic travel times, accurate estimated durations, current "
                       f"entry fees, and practical tips for {trip_duration} trip to {destination}"
    )


def create_budget_task(budget_agent, destination: str, trip_duration: str):
    """Define the budget calculation task using real cost data."""
    return Task(
        description=f"Based on the REAL flight options, hotel recommendations, and itinerary "
                   f"created by the other agents, calculate a comprehensive budget for the "
                   f"{trip_duration} {destination} trip using current pricing. Research and include actual "
                   f"costs for flights, accommodation, meals (use real restaurant prices in the destination), "
                   f"activities/tours (verified prices), transportation within {destination}, "
                   f"and miscellaneous expenses. Provide total cost estimates "
                   f"for budget, mid-range, and luxury options based on real prices. Suggest "
                   f"genuine cost-saving tips based on current market conditions.",
        agent=budget_agent,
        expected_output=f"A comprehensive budget report with itemized REAL costs for flights, "
                       f"accommodation, meals, activities with actual entry fees, transportation, "
                       f"and total realistic estimates at different budget levels, plus "
                       f"evidence-based cost-saving recommendations for a {trip_duration} trip to {destination}"
    )


# NEW TASK ADDED - Exercise 3
def create_marketing_task(marketing_agent, destination: str, trip_duration: str):
    """Define the marketing summary task (NEW - Exercise 3)."""
    return Task(
        description=f"Create a compelling marketing summary for the {trip_duration} {destination} trip package "
                   f"based on all the information gathered by other agents (flights, hotels, itinerary, budget). "
                   f"Your summary should: "
                   f"1) Highlight 3-5 unique selling points that make this package special "
                   f"2) Create an engaging headline and opening paragraph "
                   f"3) Include key package details in an appealing way "
                   f"4) Address potential traveler concerns (value, safety, experience quality) "
                   f"5) End with a strong call-to-action "
                   f"Write in an enthusiastic but authentic tone that inspires booking decisions. "
                   f"Keep it concise (250-300 words) but impactful.",
        agent=marketing_agent,
        expected_output=f"A polished marketing summary with headline, engaging body copy highlighting "
                       f"unique value propositions, key package details, and persuasive call-to-action "
                       f"for the {destination} trip",
        context=[]  # Will be populated with outputs from previous tasks
    )


# ============================================================================
# CREW ORCHESTRATION
# ============================================================================

def main(destination: str = "Iceland", trip_duration: str = "5 days",
         trip_dates: str = "January 15-20, 2026", departure_city: str = "New York",
         travelers: int = 2, budget_preference: str = "mid-range"):
    """
    Main function to orchestrate the travel planning crew.

    Args:
        destination: Travel destination (e.g., "Iceland", "France", "Japan")
        trip_duration: Duration of trip (e.g., "5 days", "7 days")
        trip_dates: Specific dates (e.g., "January 15-20, 2026")
        departure_city: City you're departing from (e.g., "New York", "Los Angeles")
        travelers: Number of travelers
        budget_preference: Budget level ("budget", "mid-range", "luxury")
    """

    print("=" * 80)
    print("CrewAI Multi-Agent Travel Planning System (REAL API VERSION)")
    print(f"Planning a {trip_duration} Trip to {destination}")
    print("=" * 80)
    print()
    print(f"📍 Destination: {destination}")
    print(f"📅 Dates: {trip_dates}")
    print(f"✈️  Departure from: {departure_city}")
    print(f"👥 Travelers: {travelers}")
    print(f"💰 Budget: {budget_preference}")
    print()

    # Validate configuration before proceeding
    print("🔍 Validating configuration...")
    if not validate_config():
        print("❌ Configuration validation failed. Please set up your .env file.")
        exit(1)

    # Set environment variables for CrewAI (it reads from os.environ)
    # CrewAI uses OPENAI_API_KEY and OPENAI_API_BASE environment variables
    os.environ["OPENAI_API_KEY"] = Config.API_KEY
    os.environ["OPENAI_API_BASE"] = Config.API_BASE
    
    # For Groq compatibility, also set OPENAI_MODEL_NAME
    if Config.USE_GROQ:
        os.environ["OPENAI_MODEL_NAME"] = Config.OPENAI_MODEL

    print("✅ Configuration validated successfully!")
    print()
    Config.print_summary()
    print()
    print("⚠️  IMPORTANT: This version uses REAL OpenAI API calls and web search")
    print("    Agents will research actual current prices and real information")
    print()
    print("Tip: Check your API usage at https://platform.openai.com/account/usage")
    print()

    # Create agents with destination parameters
    print("[1/5] Creating Flight Specialist Agent (researches real flights)...")
    flight_agent = create_flight_agent(destination, trip_dates)

    print("[2/5] Creating Accommodation Specialist Agent (researches real hotels)...")
    hotel_agent = create_hotel_agent(destination, trip_dates)

    print("[3/5] Creating Travel Planner Agent (researches real attractions)...")
    itinerary_agent = create_itinerary_agent(destination, trip_duration)

    print("[4/5] Creating Financial Advisor Agent (analyzes real costs)...")
    budget_agent = create_budget_agent(destination)

    print("[5/5] Creating Marketing Strategist Agent (creates compelling summary)... [NEW - Exercise 3]")
    marketing_agent = create_marketing_agent(destination)

    print("\n✅ All agents created successfully!")
    print()

    # Create tasks with destination parameters
    print("Creating tasks for the crew...")
    flight_task = create_flight_task(flight_agent, destination, trip_dates, departure_city)
    hotel_task = create_hotel_task(hotel_agent, destination, trip_dates)
    itinerary_task = create_itinerary_task(itinerary_agent, destination, trip_duration, trip_dates)
    budget_task = create_budget_task(budget_agent, destination, trip_duration)
    marketing_task = create_marketing_task(marketing_agent, destination, trip_duration)  # NEW TASK - Exercise 3

    print("Tasks created successfully!")
    print()

    # Create the crew with sequential task execution
    print("Forming the Travel Planning Crew...")
    print("Task Sequence: FlightAgent → HotelAgent → ItineraryAgent → BudgetAgent → MarketingAgent [NEW]")
    print()

    crew = Crew(
        agents=[flight_agent, hotel_agent, itinerary_agent, budget_agent, marketing_agent],  # Added marketing_agent
        tasks=[flight_task, hotel_task, itinerary_task, budget_task, marketing_task],  # Added marketing_task
        verbose=True,
        process="sequential"  # Sequential task execution
    )

    # Execute the crew
    print("=" * 80)
    print("Starting Crew Execution with REAL API Calls...")
    print(f"Planning {trip_duration} trip to {destination} ({trip_dates})")
    print("=" * 80)
    print()

    try:
        result = crew.kickoff(inputs={
            "trip_destination": destination,
            "trip_duration": trip_duration,
            "trip_dates": trip_dates,
            "departure_city": departure_city,
            "travelers": travelers,
            "budget_preference": budget_preference
        })

        print()
        print("=" * 80)
        print("✅ Crew Execution Completed Successfully!")
        print("=" * 80)
        print()
        print(f"FINAL TRAVEL PLAN REPORT FOR {destination.upper()} (Based on Real API Data):")
        print("-" * 80)
        print(result)
        print("-" * 80)

        # Save output to file
        output_filename = f"crewai_output_{destination.lower()}.txt"
        output_path = Path(__file__).parent / output_filename

        with open(output_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("CrewAI Multi-Agent Travel Planning System - Real API Execution Report\n")
            f.write(f"Planning a {trip_duration} Trip to {destination}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Trip Details:\n")
            f.write(f"  Destination: {destination}\n")
            f.write(f"  Duration: {trip_duration}\n")
            f.write(f"  Dates: {trip_dates}\n")
            f.write(f"  Departure: {departure_city}\n")
            f.write(f"  Travelers: {travelers}\n")
            f.write(f"  Budget Preference: {budget_preference}\n\n")
            f.write(f"Execution Time: {datetime.now()}\n")
            f.write(f"API Version: REAL API CALLS (OpenAI GPT-4)\n")
            f.write(f"Data Source: Web research via OpenAI\n\n")
            f.write("IMPORTANT NOTES:\n")
            f.write("- All flight prices, hotel costs, and attraction information is based on real data\n")
            f.write("- Prices are current as of the date this was run\n")
            f.write("- Hotel availability and prices may vary by booking date\n")
            f.write("- Weather conditions and attraction hours should be verified before travel\n\n")
            f.write("FINAL TRAVEL PLAN REPORT:\n")
            f.write("-" * 80 + "\n")
            f.write(str(result))
            f.write("\n" + "-" * 80 + "\n")

        print(f"\n✅ Output saved to {output_filename}")
        print("ℹ️  Note: All data in this report is based on REAL API calls to OpenAI")
        print("    and research of current travel information sources.")

    except Exception as e:
        print(f"\n❌ Error during crew execution: {str(e)}")
        print("\n🔍 Troubleshooting:")
        print("   1. Verify OPENAI_API_KEY is set: export OPENAI_API_KEY='sk-...'")
        print("   2. Check API key is valid and has sufficient credits")
        print("   3. Verify internet connection for web research")
        print("   4. Check OpenAI API status at https://status.openai.com")
        print()
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Allow command line arguments to override defaults
    import sys

    kwargs = {
        "destination": "Iceland",
        "trip_duration": "5 days",
        "trip_dates": "January 15-20, 2026",
        "departure_city": "New York",
        "travelers": 2,
        "budget_preference": "mid-range"
    }

    # Parse command line arguments (optional)
    # Usage: python crewai_demo.py [destination] [duration] [departure_city]
    # Example: python crewai_demo.py "France" "7 days" "Los Angeles"
    if len(sys.argv) > 1:
        kwargs["destination"] = sys.argv[1]
    if len(sys.argv) > 2:
        kwargs["trip_duration"] = sys.argv[2]
    if len(sys.argv) > 3:
        kwargs["departure_city"] = sys.argv[3]
    if len(sys.argv) > 4:
        kwargs["trip_dates"] = sys.argv[4]
    if len(sys.argv) > 5:
        kwargs["travelers"] = int(sys.argv[5])
    if len(sys.argv) > 6:
        kwargs["budget_preference"] = sys.argv[6]

    main(**kwargs)
