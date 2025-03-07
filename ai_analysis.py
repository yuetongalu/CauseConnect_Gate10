import os
import openai

# 设置OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_completion(prompt):
    """调用OpenAI API获取回复"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # 或者使用其他模型
            messages=[
                {"role": "system", "content": "You are a professional marketing expert that analyzes potential co-branding partnerships between organizations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message['content']
    except Exception as e:
        print(f"Error calling OpenAI API: {str(e)}")
        return "Unable to generate analysis at this time."

def generate_match_reasons(user_org, match_org):
    """生成匹配理由"""
    prompt = f"""
    Based on the following information, explain why these organizations would be good partners:
    
    User Organization:
    - Description: {user_org.get('Description', 'N/A')}
    - Target Audience: {user_org.get('Target Audience', 'N/A')}
    
    Potential Partner:
    - Name: {match_org['name']}
    - Description: {match_org['description']}
    - Type: {match_org['type']}
    - Industries: {match_org['industries']}
    - Specialties: {match_org['specialities']}
    
    Please provide 2-3 key points about why this would be a good partnership.
    """
    return get_completion(prompt)

def generate_match_risks(user_org, match_org):
    """生成潜在风险分析"""
    prompt = f"""
    Based on the following information, identify potential risks or challenges in this partnership:
    
    User Organization:
    - Description: {user_org.get('Description', 'N/A')}
    - Target Audience: {user_org.get('Target Audience', 'N/A')}
    
    Potential Partner:
    - Name: {match_org['name']}
    - Description: {match_org['description']}
    - Type: {match_org['type']}
    - Industries: {match_org['industries']}
    - Specialties: {match_org['specialities']}
    
    Please provide 2-3 key points about potential risks or challenges to consider.
    """
    return get_completion(prompt) 