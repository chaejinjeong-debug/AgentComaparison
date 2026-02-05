#!/usr/bin/env python3
"""Test VertexAI Sessions API using REST API directly.

This script tests the VertexAI Sessions API using direct REST calls.
"""

import asyncio
import json
from pathlib import Path

import google.auth
import google.auth.transport.requests
import httpx
from dotenv import load_dotenv

from agent_engine.envs import Env

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

# Configuration
PROJECT_ID = Env.AGENT_PROJECT_ID
LOCATION = Env.AGENT_LOCATION
AGENT_ENGINE_ID = Env.AGENT_ENGINE_ID

# API endpoints
BASE_URL = f"https://{LOCATION}-aiplatform.googleapis.com/v1"
REASONING_ENGINE = f"projects/{PROJECT_ID}/locations/{LOCATION}/reasoningEngines/{AGENT_ENGINE_ID}"


def get_access_token():
    """Get Google Cloud access token."""
    credentials, project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(google.auth.transport.requests.Request())
    return credentials.token


async def test_sessions_api():
    """Test VertexAI Sessions API."""
    print("\n" + "=" * 60)
    print("Testing VertexAI Sessions REST API")
    print("=" * 60)

    token = get_access_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. List sessions
        print("\n1. Listing sessions...")
        list_url = f"{BASE_URL}/{REASONING_ENGINE}/sessions"
        try:
            response = await client.get(list_url, headers=headers)
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                sessions = data.get("sessions", [])
                print(f"   Found {len(sessions)} sessions")
                for s in sessions[:3]:
                    print(f"     - {s.get('name', 'unknown')}")
            else:
                print(f"   Error: {response.text[:200]}")
        except Exception as e:
            print(f"   Exception: {e}")

        # 2. Create session
        print("\n2. Creating session...")
        create_url = f"{BASE_URL}/{REASONING_ENGINE}/sessions"
        session_data = {
            "userId": "test-user-rest-api",
        }
        try:
            response = await client.post(create_url, headers=headers, json=session_data)
            print(f"   Status: {response.status_code}")
            if response.status_code in [200, 201]:
                data = response.json()
                session_name = data.get("name", "")
                print(f"   Created session: {session_name}")
                return session_name
            else:
                print(f"   Error: {response.text[:300]}")
        except Exception as e:
            print(f"   Exception: {e}")

    return None


async def test_memory_api():
    """Test VertexAI Memory Bank API."""
    print("\n" + "=" * 60)
    print("Testing VertexAI Memory Bank REST API")
    print("=" * 60)

    token = get_access_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Generate memories
        print("\n1. Generating memory...")
        generate_url = f"{BASE_URL}/{REASONING_ENGINE}/memories:generate"
        memory_data = {
            "directMemoriesSource": {
                "directMemories": [{"fact": "User's name is Luke from REST API test"}]
            },
            "scope": {"userId": "test-user-rest-api"},
        }
        try:
            response = await client.post(generate_url, headers=headers, json=memory_data)
            print(f"   Status: {response.status_code}")
            if response.status_code in [200, 201]:
                data = response.json()
                print(f"   Response: {json.dumps(data, indent=2)[:200]}")
            else:
                print(f"   Error: {response.text[:300]}")
        except Exception as e:
            print(f"   Exception: {e}")

        # 2. Retrieve memories
        print("\n2. Retrieving memories...")
        retrieve_url = f"{BASE_URL}/{REASONING_ENGINE}/memories:retrieve"
        retrieve_data = {"scope": {"userId": "test-user-rest-api"}}
        try:
            response = await client.post(retrieve_url, headers=headers, json=retrieve_data)
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                memories = data.get("memories", [])
                print(f"   Found {len(memories)} memories")
                for m in memories[:3]:
                    print(f"     - {m.get('fact', 'unknown')[:50]}")
            else:
                print(f"   Error: {response.text[:300]}")
        except Exception as e:
            print(f"   Exception: {e}")


async def check_api_availability():
    """Check which API endpoints are available."""
    print("\n" + "=" * 60)
    print("Checking API Endpoint Availability")
    print("=" * 60)

    token = get_access_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    endpoints_to_check = [
        ("GET", f"{BASE_URL}/{REASONING_ENGINE}", "Reasoning Engine Info"),
        ("GET", f"{BASE_URL}/{REASONING_ENGINE}/sessions", "List Sessions"),
        ("POST", f"{BASE_URL}/{REASONING_ENGINE}/sessions", "Create Session"),
        ("POST", f"{BASE_URL}/{REASONING_ENGINE}/memories:retrieve", "Retrieve Memories"),
        ("POST", f"{BASE_URL}/{REASONING_ENGINE}/memories:generate", "Generate Memories"),
    ]

    async with httpx.AsyncClient(timeout=30.0) as client:
        for method, url, name in endpoints_to_check:
            try:
                if method == "GET":
                    response = await client.get(url, headers=headers)
                else:
                    response = await client.post(url, headers=headers, json={})

                status = "AVAILABLE" if response.status_code < 500 else "ERROR"
                print(f"  {name}: {response.status_code} ({status})")

                if response.status_code >= 400:
                    error_msg = response.text[:100].replace("\n", " ")
                    print(f"    -> {error_msg}")

            except Exception as e:
                print(f"  {name}: EXCEPTION - {e}")


async def main():
    """Run API tests."""
    print("=" * 60)
    print("VertexAI REST API Integration Tests")
    print("=" * 60)
    print(f"Project: {PROJECT_ID}")
    print(f"Location: {LOCATION}")
    print(f"Agent Engine: {AGENT_ENGINE_ID}")
    print(f"Base URL: {BASE_URL}")

    # Check API availability
    await check_api_availability()

    # Test Sessions API
    await test_sessions_api()

    # Test Memory API
    await test_memory_api()

    print("\n" + "=" * 60)
    print("Tests completed")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
