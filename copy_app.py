import openai
import json
import pandas as pd
from pymongo import MongoClient
import numpy as np
from bson.binary import Binary
import streamlit as st
from scipy.spatial.distance import cosine
import requests
from dotenv import load_dotenv
import os
import certifi
import base64
from ai_analysis import generate_match_reasons, generate_match_risks

# ----------- Streamlit UI代码 -----------
# 设置页面配置（必须是第一个Streamlit命令）
st.set_page_config(page_title="Organization Matcher", layout="wide")

# 添加清除环境变量缓存的功能
def reload_env():
    """重新加载环境变量"""
    # 清除特定的环境变量
    env_vars = [
        "MATCH_EVALUATION_SYSTEM_PROMPT",
        "MATCH_EVALUATION_PROMPT",
        "MONGODB_URI",
        "MONGODB_DB_NAME",
        "MONGODB_COLLECTION_NONPROFIT",
        "MONGODB_COLLECTION_FORPROFIT",
        "OPENAI_API_KEY"
    ]
    
    # 清除指定的环境变量
    for var in env_vars:
        if var in os.environ:
            del os.environ[var]
    
    # 重新加载.env文件
    load_dotenv(override=True)
    return "环境变量已重新加载"

# 在侧边栏添加重新加载按钮
with st.sidebar:
    if st.button("🔄 重新加载环境变量"):
        result = reload_env()
        st.success(result)
        st.rerun()

# ----------- Load Environment Variables -----------
load_dotenv()

# ----------- MongoDB Connection -----------
try:
    # 连接MongoDB
    client = MongoClient(os.getenv("MONGODB_URI"), tlsCAFile=certifi.where())
    # 测试连接
    client.server_info()
    
    # 获取数据库
    db = client[os.getenv("MONGODB_DB_NAME")]
    
    # 初始化集合，使用环境变量中定义的集合名称
    collection1 = db[os.getenv("MONGODB_COLLECTION_NONPROFIT")]  # 非营利组织集合
    collection2 = db[os.getenv("MONGODB_COLLECTION_FORPROFIT")]  # 营利组织集合
    
    # 在侧边栏显示连接信息
    with st.sidebar:
        with st.expander("Database Connection Status", expanded=False):
            st.success("Successfully connected to MongoDB")
            st.write(f"Connected to database: {os.getenv('MONGODB_DB_NAME')}")
            st.write(f"Number of documents in {os.getenv('MONGODB_COLLECTION_NONPROFIT')}: {collection1.count_documents({})}")
            st.write(f"Number of documents in {os.getenv('MONGODB_COLLECTION_FORPROFIT')}: {collection2.count_documents({})}")
    
except Exception as e:
    # 错误信息仍然在主页面显示，因为这是重要的错误提示
    st.error(f"Error connecting to MongoDB: {str(e)}")
    raise Exception(f"Failed to connect to MongoDB: {str(e)}")

# 设置OpenAI API密钥
openai.api_key = os.getenv("OPENAI_API_KEY")

# ----------- 函数定义 -----------
def generate_ideal_organization(row):
    """Generate 10 organizations based on needs, then filter to 3 based on mission alignment."""
    try:
        # Step 1: Generate 10 potential organizations
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": os.getenv("PROMPT_GEN_ORG_SYSTEM").format(
                    org_type_looking_for=row["Organization looking 1"])},
                {"role": "user", "content": os.getenv("PROMPT_GEN_ORG_USER").format(
                    org_type_looking_for=row["Organization looking 1"],
                    partnership_description=row["Organization looking 2"])}
            ]
        )

        generated_organizations = response['choices'][0]['message']['content'].strip()

        # Step 2: Filter down to the 3 best matches
        filtered_response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": os.getenv("PROMPT_FILTER_SYSTEM")},
                {"role": "user", "content": os.getenv("PROMPT_FILTER_USER").format(
                    organization_mission=row["Description"],
                    generated_organizations=generated_organizations)}
            ]
        )

        return filtered_response['choices'][0]['message']['content'].strip()

    except Exception as e:
        st.error(f"Error generating organizations: {str(e)}")
        return ""

# ----------- Define Structured Tagging Steps -----------
step_descriptions = {
    1: os.getenv("TAG_STEP_1"),
    2: os.getenv("TAG_STEP_2"),
    3: os.getenv("TAG_STEP_3"),
    4: os.getenv("TAG_STEP_4"),
    5: os.getenv("TAG_STEP_5"),
    6: os.getenv("TAG_STEP_6")
}

def generate_fixed_tags(description, audience, total_tags=30, steps=6, tags_per_step=5):
    """Generate structured AI-powered tags following a 6-step format."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # 使用相同的模型
            messages=[
                {"role": "system", "content": os.getenv("PROMPT_TAGS_SYSTEM").format(
                    total_tags=total_tags,
                    steps=steps,
                    tags_per_step=tags_per_step
                )},
                {"role": "user", "content": os.getenv("PROMPT_TAGS_USER").format(
                    total_tags=total_tags,
                    description=description
                )}
            ]
        )
        tags = response['choices'][0]['message']['content'].strip()

        # Convert tags to a list and normalize to exactly `total_tags`
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        tag_list = tag_list[:total_tags]  # Ensure 30 tags

        return ", ".join(tag_list)
    except Exception as e:
        st.error(f"Error generating tags: {str(e)}")
        return None

def get_embedding(text):
    """Generate vector embedding using OpenAI."""
    if not text or not isinstance(text, str):
        return None  # Skip invalid data

    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=[text]
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        return None

def find_top_50_matches(embedding, looking_for_type):
    """Find top 50 matching organizations based on embedding similarity."""
    matches = []
    # 确保类型匹配完全一致
    collection = collection1 if looking_for_type.strip() == os.getenv("MONGODB_COLLECTION_NONPROFIT").strip() else collection2
    
    try:
        st.write(f"Searching in {looking_for_type} database...")
        for org in collection.find({"Embedding": {"$exists": True}}):
            if org.get("Embedding"):
                # Convert from BSON Binary to numpy array
                org_embedding = np.frombuffer(org["Embedding"], dtype=np.float32)
                # Calculate similarity
                similarity = 1 - cosine(embedding, org_embedding)
                matches.append((
                    similarity,
                    org.get("Name", "Unknown"),
                    org.get("Description", "No description available"),
                    org.get("URL", "N/A"),
                    org.get("linkedin_description", "No LinkedIn description available"),
                    org.get("linkedin_tagline", "No tagline available"),
                    org.get("linkedin_type", "N/A"),
                    org.get("linkedin_industries", "N/A"),
                    org.get("linkedin_specialities", "N/A"),
                    org.get("linkedin_staff_count", "N/A"),
                    org.get("City", "N/A"),
                    org.get("State", "N/A"),
                    org.get("linkedin_url", "N/A"),
                    org.get("Tag", "No tags available")
                ))
        
        matches.sort(reverse=True)
        st.write(f"Found {len(matches)} potential matches")
        return matches[:100]  # Return top 50 instead of 100
    except Exception as e:
        st.error(f"Error finding matches: {str(e)}")
        return []

def evaluate_match_with_gpt(org_info, user_info):
    """Use GPT to evaluate match quality and decide whether to keep the match"""
    try:
        # Get prompts from environment variables
        system_prompt = os.getenv("MATCH_EVALUATION_SYSTEM_PROMPT")
        evaluation_prompt = os.getenv("MATCH_EVALUATION_PROMPT")
        
        if not system_prompt or not evaluation_prompt:
            raise ValueError("Required prompts not found in environment variables")

        # Format the evaluation prompt
        try:
            formatted_prompt = evaluation_prompt.format(
                user_name=user_info['Name'],
                user_type=user_info['Type'],
                user_description=user_info['Description'],
                user_target_audience=user_info['Target Audience'],
                user_looking_type=user_info['Organization looking 1'],
                user_partnership_desc=user_info['Organization looking 2'],
                match_name=org_info[1],
                match_description=org_info[2],
                match_linkedin_desc=org_info[4],
                match_tagline=org_info[5],
                match_type=org_info[6],
                match_industry=org_info[7],
                match_specialties=org_info[8],
                match_tags=org_info[13]
            )
        except KeyError as e:
            st.error(f"Missing required field in organization data: {str(e)}")
            return False
        except Exception as e:
            st.error(f"Error formatting prompt: {str(e)}")
            return False

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0.3
        )
        
        result = response['choices'][0]['message']['content'].strip().lower()
        return result == 'true'
    except Exception as e:
        st.error(f"Error evaluating match: {str(e)}")
        return False  # Changed to False to be more conservative with errors

def process_matches(tags, looking_for_type, row):
    """处理匹配逻辑，确保返回20个匹配结果"""
    if not tags:
        return []
    
    embedding = get_embedding(tags)
    if embedding is None:
        return []
    
    # 获取前100个相似度匹配
    all_matches = find_top_50_matches(embedding, looking_for_type)
    if not all_matches:
        return []
    
    st.write("Using AI to analyze match quality in depth...")
    filtered_matches = []
    similarity_matches = []
    
    with st.spinner("Analyzing matches..."):
        progress_bar = st.progress(0)
        
        # 对前30个进行GPT评估
        for i, match in enumerate(all_matches[:30]):
            if evaluate_match_with_gpt(match, row):
                filtered_matches.append(match)
            progress_bar.progress((i + 1) / 30)
        
        # 情况1：如果有20个或更多GPT评估成功结果，只返回前20个AI评估匹配
        if len(filtered_matches) >= 20:
            return filtered_matches[:20]
            
        # 情况2：如果有一些但不足20个GPT评估结果，补充相似度匹配
        elif len(filtered_matches) > 0:
            similarity_matches = [match for match in all_matches if match not in filtered_matches]
            needed_similarity_matches = 20 - len(filtered_matches)
            return filtered_matches + similarity_matches[:needed_similarity_matches]
            
        # 情况3：如果没有GPT评估通过的结果，返回前20个相似度匹配
        else:
            return all_matches[:20]

def display_matches(filtered_matches, gpt_verified_count):
    """显示匹配结果，包含公司logo"""
    st.subheader("Top 20 Matching Organizations:")
    st.write(f"({gpt_verified_count} AI-verified matches, {20-gpt_verified_count} Similarity-based matches)")
    
    for i, match in enumerate(filtered_matches, 1):
        similarity, name, description, url, linkedin_desc, tagline, type_, industries, specialities, staff_count, city, state, linkedin_url, match_tags = match
        match_type = "AI-Verified Match" if i <= gpt_verified_count else "Similarity Match"
        
        with st.expander(f"{i}. {name} ({match_type}, Similarity: {match[0]:.2%})"):
            # 创建主要内容区域
            cols = st.columns([1.5, 2, 1])
            
            # 左侧列：Logo和基本联系信息
            with cols[0]:
                # Logo区域
                try:
                    collection = collection1 if looking_for_type.strip() == os.getenv("MONGODB_COLLECTION_NONPROFIT").strip() else collection2
                    org_data = collection.find_one({"Name": name})
                    if org_data and "linkedin_logo" in org_data:
                        logo_url = org_data["linkedin_logo"]
                        st.image(logo_url, width=150)
                    else:
                        st.markdown("Company Logo", unsafe_allow_html=True)
                except Exception as e:
                    st.markdown("Company Logo", unsafe_allow_html=True)
                
                st.markdown("### Tagline")
                if tagline and tagline != "N/A":
                    st.markdown(f"*\"{tagline}\"*")

                # 基本信息
                st.markdown("### Location")
                st.markdown(f"📍 {city}, {state}" if city and state else "📍 Location N/A")
                
                st.markdown("### Website")
                if url and url != "N/A":
                    st.markdown(f"🌐 [{url}]({url})")

                
                st.markdown("### LinkedIn")
                if linkedin_url and linkedin_url != "N/A":
                    st.markdown(f"[<img src='https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Logo.svg.original.svg' width='80'> View Profile]({linkedin_url})", unsafe_allow_html=True)

                
                st.markdown("### Staff Count")
                st.markdown(f"👥 {staff_count}" if staff_count and staff_count != "N/A" else "👥 N/A")
            
            # 中间列：主要信息
            with cols[1]:
                st.markdown("### Basic Information")
                st.write(description if description else "No description available...")
                
                st.markdown("### Type & Industry")
                st.markdown(f"**Type**  \n{type_}" if type_ else "**Type**  \nN/A")
                st.markdown(f"**Industry**  \n{industries}" if industries else "**Industry**  \nN/A")
            
            # 右侧列：特色和LinkedIn信息
            with cols[2]:
                st.markdown("### Specialties")
                if specialities and specialities != "N/A":
                    specialties_list = [s.strip() for s in specialities.split(',')]
                    for specialty in specialties_list:
                        st.markdown(f'<span style="background-color: #f0f2f6; padding: 5px 10px; border-radius: 15px; margin: 2px; display: inline-block; font-size: 0.85em;">{specialty}</span>', unsafe_allow_html=True)
                
               
            
            # 添加分隔线
            st.divider()
            
            # AI Reasoning 部分
            st.markdown("### 🤖 AI Match Analysis")
            
            # 创建两列布局用于显示匹配理由和风险
            reason_col1, reason_col2 = st.columns(2)
            
            with reason_col1:
                st.markdown("#### ✨ Why We Think This is a Good Match")
                # 使用 session_state 来保存分析结果
                if f"match_reasons_{i}" not in st.session_state:
                    with st.spinner("Analyzing partnership potential..."):
                        try:
                            st.session_state[f"match_reasons_{i}"] = generate_match_reasons(row, {
                                "name": name,
                                "description": description,
                                "type": type_,
                                "industries": industries,
                                "specialities": specialities
                            })
                        except Exception as e:
                            st.error("Unable to generate match analysis.")
                
                # 显示分析结果
                if st.session_state[f"match_reasons_{i}"]:
                    st.write(st.session_state[f"match_reasons_{i}"])
            
            with reason_col2:
                st.markdown("#### ⚠️ Potential Risks & Considerations")
                # 使用 session_state 来保存风险分析结果
                if f"match_risks_{i}" not in st.session_state:
                    with st.spinner("Analyzing potential risks..."):
                        try:
                            st.session_state[f"match_risks_{i}"] = generate_match_risks(row, {
                                "name": name,
                                "description": description,
                                "type": type_,
                                "industries": industries,
                                "specialities": specialities
                            })
                        except Exception as e:
                            st.error("Unable to generate risk analysis.")
                
                # 显示风险分析结果
                if st.session_state[f"match_risks_{i}"]:
                    st.write(st.session_state[f"match_risks_{i}"])

# ----------- Streamlit UI代码 -----------
# 获取图片路径
image_path = "resource/pexels-adrien-olichon-1257089-3137052.jpg"

# 将图片转换为 base64
def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# 获取图片的 base64 编码
img_base64 = get_base64_of_image(image_path)

# 添加背景图片的CSS
st.markdown(f"""
<style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
    }}

    /* 添加半透明白色遮罩 */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(255, 255, 255, 0.5);  /* 白色背景，50%透明度 */
        z-index: -1;
    }}
</style>
""", unsafe_allow_html=True)

# 创建主标题和介绍
st.title("Organization Partnership Matcher")
st.markdown("""
This tool helps you find potential partnership organizations that align with your values and goals.
Please fill in the information about your organization below.
""")

# 添加分割线
st.divider()  # 或者使用 st.markdown("---")

# 第一部分：组织档案
st.header(" 1️⃣  Build Your Organization Profile")

# Create two columns with equal width
col1, col2 = st.columns(2)

# Place org_name in the first column
with col1:
    org_name = st.text_input("Organization Name*", key="name")

# Place org_type in the second column
with col2:
    org_type = st.selectbox(
        "Your Organization Type*",
        ["Non Profit", "For-Profit"],
        key="type"
    )

org_description = st.text_area(
    "Organization Mission Statement*",
    help="What is your organization's mission?",
    placeholder="Describe your organization's purpose and goals (e.g., \"Our mission is to promote sustainable living through education and community-driven initiatives.\")",
    key="description"
)

# 创建两列布局
col1, col2 = st.columns(2)

# 左列：核心价值观输入
with col1:
    org_category = st.text_area(
        "Core Values*",
        help="What are the top three core values your brand stands for?",
        placeholder="List the top three fundamental principles that guide your organization (e.g., \"Sustainability, Inclusivity, Innovation, Transparency\").",
        key="category"
    )

# 右列：目标受众输入
with col2:
    target_audience = st.text_area(
        "Target Audience*",
        help="Who does your company serve? What social causes does your customer base care about?",
        placeholder="Who does your organization serve? (e.g., \"Young professionals interested in environmental conservation, local communities, small businesses seeking sustainable solutions\").",
        key="audience"
    )

# 组合所有描述性信息为一个完整的描述
combined_description = f"""Organization Mission:
{org_description}

Core Values:
{org_category}

Target Audience:
{target_audience}"""

# 位置信息
col3, col4 = st.columns(2)
with col3:
    state = st.text_input("State", key="state")
with col4:
    city = st.text_input("City", key="city")

website = st.text_input("Website URL", key="website")

# 添加分割线
st.divider()  # 或者使用 st.markdown("---")

# 第二部分：匹配过程
st.header(" 2️⃣  Start the Matching Process")

looking_for_type = st.selectbox(
    "Organization Type Looking For*",
    ["Non Profit", "For-Profit"],
    key="looking_for_type"
)

looking_for_description = st.text_area(
    "What Kind of Organization Are You Looking For?*",
    help="E.g.: A nonprofit that supports financial literacy for young adults, as our company provides budgeting and investment tools designed for first-time investors.",
    placeholder="Describe the type of partner you seek (e.g., \"Environmental NGOs focused on climate action,\" \"Tech companies interested in corporate social responsibility,\" \"Local businesses supporting green initiatives\").",
    key="looking_for_desc"
)

# 添加一个分隔线在按钮前
st.divider()

# 添加自定义CSS样式
st.markdown("""
<style>
    .stButton > button {
        background-color: #93C7C1;
        color: white;
    }
    .stButton > button:hover {
        background-color: #7BA8A3;  /* 悬停时的颜色稍深 */
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# 使用列布局来使按钮居中且更大
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    button_clicked = st.button("Find Matching Organizations 🔍 ", type="primary", use_container_width=True)

# 按钮点击后的逻辑移出列布局
if button_clicked:
    if not all([org_name, org_type, org_description, org_category, target_audience, looking_for_type, looking_for_description]):
        st.error("Please fill in all required fields marked with *")
    else:
        row = {
            "Name": org_name,
            "Type": org_type,
            "Description": combined_description,
            "State": state,
            "City": city,
            "URL": website,
            "Organization looking 1": looking_for_type,
            "Organization looking 2": looking_for_description,
            "Target Audience": target_audience
        }
        
        with st.spinner("Finding matching organizations..."):
            generated_orgs = generate_ideal_organization(pd.Series(row))
            st.subheader("Suggested Organizations:")
            st.write(generated_orgs)
            
            tags = generate_fixed_tags(generated_orgs, row["Target Audience"])
            st.subheader("Generated Tags:")
            st.write(tags)
            
            if tags:
                filtered_matches = process_matches(tags, looking_for_type, row)
                display_matches(filtered_matches, len(filtered_matches))

# 在侧边栏中显示环境变量状态
with st.sidebar:
    # 放在数据库连接状态的下方
    with st.expander("Environment Variables Status", expanded=False):
        for var in ["MATCH_EVALUATION_SYSTEM_PROMPT", "MATCH_EVALUATION_PROMPT"]:
            value = os.getenv(var)
            st.write(f"{var}: {'Found' if value else 'Missing'}")

# Verify required environment variables
required_env_vars = [
    "OPENAI_API_KEY",
    "MONGODB_URI",
    "MONGODB_DB_NAME"
]

# Check if all required environment variables are present
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

