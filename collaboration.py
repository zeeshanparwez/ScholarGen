import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
import json
from langchain_core.messages import HumanMessage

PROFILE_PATH = 'Data/user_profiles.xlsx'
embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import json

# Load environment variables from your Config/.env
load_dotenv("./Config/.env")

# Initialize Gemini LLM client
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    max_retries=2
)

# Initialize LLM (place at module level)
load_dotenv("./Config/.env")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    max_retries=2
)

def extract_profile_from_text(text: str) -> dict:
    """
    Fixed version using llm.invoke() and robust JSON parsing.
    """
    prompt = (
        "Extract user's interests and skills from the following text. "
        "Return ONLY a valid JSON dictionary with keys 'interests' and 'skills'.\n\n"
        f"Text:\n{text}\n\n"
        "Respond with ONLY this format:\n"
        '{"interests": ["AI", "Machine Learning"], "skills": ["Python", "Data Analysis"]}'
    )
    
    try:
        # Use invoke() instead of deprecated predict()
        message = HumanMessage(content=prompt)
        response = llm.invoke([message])
        
        # Get content and clean it
        response_text = response.content.strip()
        
        # Try to extract JSON from response (handle extra text)
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx != 0:
            json_str = response_text[start_idx:end_idx]
            profile = json.loads(json_str)
        else:
            profile = json.loads(response_text)
        
        # Ensure required keys exist
        profile.setdefault('interests', [])
        profile.setdefault('skills', [])
        
        return profile
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
        print(f"Raw response: {response.content[:200]}...")
        return {"interests": [], "skills": []}
    except Exception as e:
        print(f"Profile extraction failed: {e}")
        return {"interests": [], "skills": []}


'''def extract_profile_from_text(text: str) -> dict:
    """
    Calls Gemini LLM to extract user profile information such as interests and skills.
    
    Args:
        text (str): The user input text or conversation content to analyze.
    
    Returns:
        dict: Dictionary with keys 'interests' and 'skills', values are lists of extracted items.
    """
    prompt = (
        "Extract user's interests and skills from the following text. "
        "Return the result as a JSON dictionary with keys 'interests' and 'skills'.\n\n"
        f"Text:\n{text}\n\n"
        "Output format example: {\"interests\": [\"AI\", \"Machine Learning\"], \"skills\": [\"Python\", \"Data Analysis\"]}"
    )
    try:
        # Call Gemini LLM with the prompt
        response = llm.predict(prompt)
        # Parse JSON response
        profile = json.loads(response)
        # Validate required keys
        if not ("interests" in profile and "skills" in profile):
            raise ValueError("Response missing required keys")
        return profile
    except Exception as e:
        print(f"Profile extraction failed: {e}")
        return {"interests": [], "skills": []}
'''


def load_profiles():
    if not os.path.exists(PROFILE_PATH):
        return pd.DataFrame(columns=['username', 'interests', 'skills', 'last_updated'])
    return pd.read_excel(PROFILE_PATH)

def compute_profile_embeddings(df):
    # Combine interests and skills into a single text string for embedding
    profile_texts = df.apply(
        lambda r: ' '.join(eval(r['interests'])) + ' ' + ' '.join(eval(r['skills'])) if pd.notna(r['interests']) else '',
        axis=1
    ).tolist()
    return embedding_model.encode(profile_texts, convert_to_numpy=True)

def match_similar_users(username: str, top_n=3):
    df = load_profiles()
    if username not in df['username'].values:
        return []

    embeddings = compute_profile_embeddings(df)
    user_index = df.index[df['username'] == username][0]
    similarities = cosine_similarity([embeddings[user_index]], embeddings)[0]

    # Sort users by similarity, exclude the user itself (highest similarity)
    similar_indices = similarities.argsort()[::-1]
    similar_indices = [idx for idx in similar_indices if idx != user_index]

    top_indices = similar_indices[:top_n]
    return df.iloc[top_indices]['username'].tolist()

'''def suggest_collaboration_topics(usernames):
    from course_retriever import CourseRetriever
    from research_mcp import search_papers

    # Initialize retriever objects (adjust filepath/env variables accordingly)
    MAX_RESULTS = 3
    try:
        course_retriever = CourseRetriever(filepath=os.getenv("COURSEDATAPATH"))
    except Exception:
        course_retriever = None

    topics = []
    profiles_df = load_profiles()
    interests_set = set()

    for user in usernames:
        user_row = profiles_df[profiles_df['username'] == user]
        if not user_row.empty:
            interests = eval(user_row.iloc[0]['interests'])
            interests_set.update(interests)

    interests_list = list(interests_set)[:MAX_RESULTS]

    for interest in interests_list:
        if course_retriever:
            try:
                courses = course_retriever.get_top_courses(interest, MAX_RESULTS)
                for course in courses:
                    topics.append(f"Course: {course['title']} - Consider group study or project")
            except Exception:
                pass

        try:
            papers = searchpapers(interest, MAX_RESULTS)
            for paper in papers:
                topics.append(f"Research Paper: {paper['title']} - Discuss findings and applications")
        except Exception:
            pass

    if not topics:
        topics.append("No specific collaboration topics found. Explore new areas together!")

    return topics
'''
def suggest_collaboration_topics(usernames: list) -> list:
    """
    Simple collaboration topics based only on user_profiles data.
    """
    topics = []
    
    # Load profiles
    profiles_df = load_profiles()
    
    # Aggregate interests and skills from matched users
    all_interests = set()
    all_skills = set()
    
    for username in usernames:
        user_row = profiles_df[profiles_df['username'] == username]
        if not user_row.empty:
            try:
                interests = eval(user_row.iloc[0]['interests']) if pd.notna(user_row.iloc[0]['interests']) else []
                skills = eval(user_row.iloc[0]['skills']) if pd.notna(user_row.iloc[0]['skills']) else []
                
                all_interests.update(interests)
                all_skills.update(skills)
            except:
                continue
    
    # Simple collaboration topics only
    if all_interests:
        topics.append(f"Interests: {', '.join(list(all_interests)[:3])}")
    
    if all_skills:
        topics.append(f"Skills: {', '.join(list(all_skills)[:3])}")
    
    # Study group with usernames
    if usernames:
        topics.append(f"Study group: {', '.join(usernames)}")
    
    return topics

def save_profiles(df: pd.DataFrame) -> None:
    """
    Save profiles DataFrame to Excel with intelligent handling:
    - Creates directory if it doesn't exist
    - For NEW users: Appends new row
    - For EXISTING users: Updates their row only
    - Preserves all other users' data
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(PROFILE_PATH)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # If file doesn't exist, create it with the DataFrame
    if not os.path.exists(PROFILE_PATH):
        df.to_excel(PROFILE_PATH, index=False)
        return

    # File exists - load existing data
    existing_df = pd.read_excel(PROFILE_PATH)
    
    # For each new row in df (new users), append to existing
    new_rows = df[~df['username'].isin(existing_df['username'])]
    if not new_rows.empty:
        updated_df = pd.concat([existing_df, new_rows], ignore_index=True)
    else:
        updated_df = existing_df.copy()

    # Update existing users' data
    for username in df['username'].values:
        if username in updated_df['username'].values:
            mask = updated_df['username'] == username
            updated_df.loc[mask, ['interests', 'skills', 'last_updated']] = \
                df.loc[df['username'] == username, ['interests', 'skills', 'last_updated']].values[0]

    # Save the final DataFrame
    updated_df.to_excel(PROFILE_PATH, index=False)

def update_user_profile_with_timestamp(username, profile, timestamp):
    df = load_profiles()
    if username in df['username'].values:
        df.loc[df['username'] == username, ['interests', 'skills', 'last_updated']] = [
            str(profile.get('interests', [])),
            str(profile.get('skills', [])),
            timestamp
        ]
    else:
        new_row = {
            'username': username,
            'interests': str(profile.get('interests', [])),
            'skills': str(profile.get('skills', [])),
            'last_updated': timestamp
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_profiles(df)

if __name__ == "__main__":
    '''profile = extract_profile_from_text("Search for different research papers on Machine Learning")
    print(f"Profile: {profile}")
    timestamp = datetime.datetime.now().isoformat()
        
    # Extend update_user_profile to also save timestamp
    update_user_profile_with_timestamp(
        username="test_user_1",
        profile=profile,
        timestamp=timestamp
    )'''
    #print(match_similar_users('test_user'))
    matched_users = match_similar_users('test_user')
    print(f"Matched Users: {matched_users}")
    collaboration_topics = suggest_collaboration_topics(matched_users)
    print(f"Collaboration Topics: {collaboration_topics}")
