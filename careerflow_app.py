import gradio as gr
import google.generativeai as genai
import json
import os

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    api_key = "YOUR_GEMINI_API_KEY"  # Fallback for testing

genai.configure(api_key=api_key)

# === CANDIDATE AGENT ===
def candidate_agent(job_posting, resume_text):
    """Tailors resume to job posting requirements"""
    if not job_posting or not resume_text:
        return "❌ Please fill both job posting and resume fields."
    
    prompt = f"""You are an expert career coach. Analyze this job posting and resume.

JOB POSTING:
{job_posting}

CANDIDATE RESUME:
{resume_text}

Your task:
1. Identify 5 KEY REQUIREMENTS from the job posting
2. Map the candidate's experience to those requirements
3. Suggest specific resume edits to highlight relevant skills/projects
4. Identify missing keywords to add naturally
5. Provide a tailored version of the resume

Format your response as:
KEY REQUIREMENTS:
- [List 5 key requirements]

SKILL MAPPING:
- [How candidate's experience maps to requirements]

SUGGESTED EDITS:
- [Specific resume edits]

TAILORED RESUME:
[Generate improved resume text]"""

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error: {str(e)}"

# === RECRUITER AGENT ===
def recruiter_agent(resume_text, job_requirements):
    """Scores candidate fit and provides reasoning"""
    if not resume_text or not job_requirements:
        return json.dumps({"error": "Please fill both resume and job requirements fields."}, indent=2)
    
    prompt = f"""You are an expert recruiter evaluating candidates.

RESUME:
{resume_text}

JOB REQUIREMENTS:
{job_requirements}

Your task:
1. Analyze skills match (must-haves vs nice-to-haves)
2. Evaluate experience level (junior/mid/senior)
3. Assess project relevance
4. Identify red flags or concerns
5. Score overall fit from 0-100

Return a JSON response with this exact format:
{{
    "fit_score": <0-100 number>,
    "experience_level": "<junior/mid/senior>",
    "skills_match": {{
        "must_haves": "<% match>",
        "nice_to_haves": "<% match>"
    }},
    "strengths": ["<top 3 strengths>"],
    "gaps": ["<top 2 skill gaps>"],
    "red_flags": ["<any concerns or null if none>"],
    "reasoning": "<brief explanation of score>"
}}"""

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        
        # Extract JSON from response
        response_text = response.text
        # Try to parse as JSON
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                return json.dumps(result, indent=2)
        except:
            pass
        
        # If not valid JSON, return as formatted text
        return response_text
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

# === COMBINED AGENT ===
def full_careerflow(job_posting, resume_text, mode):
    """Run either candidate or recruiter agent"""
    if mode == "Candidate (Tailor Resume)":
        return candidate_agent(job_posting, resume_text)
    else:
        return recruiter_agent(resume_text, job_posting)

# === GRADIO INTERFACE ===
with gr.Blocks(title="CareerFlow - AI Hiring Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚀 CareerFlow: AI Agent for Intelligent Hiring
    
    **Two-sided AI agent that reasons about job-candidate fit**
    
    - **Candidate Agent**: Tailor your resume to any job posting
    - **Recruiter Agent**: Score candidate fit with reasoning
    
    Built with Gemini 2.0 Flash | Kaggle AI Agents Capstone
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📋 Select Mode")
            mode = gr.Radio(
                ["Candidate (Tailor Resume)", "Recruiter (Score Fit)"],
                value="Candidate (Tailor Resume)",
                label="Agent Mode"
            )
        
        with gr.Column():
            gr.Markdown("### ℹ️ How It Works")
            gr.Markdown("""
            **Candidate Mode**: 
            - Input: Job posting + your resume
            - Output: Tailored resume with suggestions
            
            **Recruiter Mode**:
            - Input: Resume + job requirements
            - Output: Fit score (0-100) + reasoning
            """)
    
    gr.Divider()
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📝 Input")
            job_posting = gr.Textbox(
                label="Job Posting / Job Requirements",
                placeholder="Paste job description or requirements here...",
                lines=8,
                value="DRDO ML Engineer. Skills: Python, PyTorch, Computer Vision. Experience: 2+ years. Education: B.Tech/M.Tech."
            )
            
            resume_text = gr.Textbox(
                label="Resume / Candidate Profile",
                placeholder="Paste your resume or candidate profile here...",
                lines=8,
                value="Final-year B.Tech in AI & ML. CGPA: 8.39. Projects: AgriSense (IoT+ML), PrepGrid (interview tracker). Skills: Python, FastAPI, Gradio, LangChain, PyTorch. GitHub: harisaravananm"
            )
            
            submit_btn = gr.Button("🔍 Run Agent", size="lg", variant="primary")
        
        with gr.Column():
            gr.Markdown("### 📊 Output")
            output = gr.Textbox(
                label="Agent Response",
                lines=15,
                interactive=False,
                show_copy_button=True
            )
    
    gr.Divider()
    
    # Examples section
    with gr.Row():
        gr.Markdown("""
        ### 💡 Example Inputs
        
        **Candidate Mode Example:**
        - Job: "Senior ML Engineer at Google. Python, TensorFlow, 5+ years exp."
        - Resume: "B.Tech graduate, 2 years ML experience, Python, PyTorch"
        
        **Recruiter Mode Example:**
        - Resume: "Full-stack developer, 3 years exp, Node.js, React, Docker"
        - Job: "Startup CTO role, needs: Python, ML, 5+ years"
        """)
    
    # Connect button to function
    submit_btn.click(
        fn=full_careerflow,
        inputs=[job_posting, resume_text, mode],
        outputs=output
    )
    
    gr.Markdown("""
    ---
    **GitHub**: [harisaravananm/careerflow-agent](https://github.com/harisaravananm/careerflow-agent)
    
    **Kaggle Submission**: Vibecoding Agents Capstone Project
    """)

if __name__ == "__main__":
    demo.launch()
