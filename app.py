# language: Python
# filepath: /c:/Users/tmina/github/open_deep_research/app.py
import streamlit as st
import threading
import os
from dotenv import load_dotenv
# from huggingface_hub import login
from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SearchInformationTool,
    SimpleTextBrowser,
    VisitTool,
)
from scripts.visual_qa import visualizer

from smolagents import (
    CodeAgent,
    LiteLLMModel,
    ToolCallingAgent,
)

AUTHORIZED_IMPORTS = [
    "requests",
    "zipfile",
    "os",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "pubchempy",
    "xml",
    "yahoo_finance",
    "Bio",
    "sklearn",
    "scipy",
    "pydub",
    "io",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "torch",
    "datetime",
    "fractions",
    "csv",
]
load_dotenv(override=True)
# login(os.getenv("HF_TOKEN"))

append_answer_lock = threading.Lock()

custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}
os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)

def translate_to_japanese(text, model):
    translation_prompt = (
        "Translate the following text to Japanese. Please output only the Japanese translation, with no additional explanation or commentary:\n\n" + text
    )
    messages = [{"role": "user", "content": translation_prompt}]
    translated_text = model(messages)
    return translated_text if isinstance(translated_text, str) else translated_text.content

@st.cache_data(show_spinner=False)
def run_open_deep_research(question, model_id="o1"):
    text_limit = 100000

    model = LiteLLMModel(
        model_id,
        custom_role_conversions=custom_role_conversions,
        max_completion_tokens=8192,
    )
    document_inspection_tool = TextInspectorTool(model, text_limit)

    browser = SimpleTextBrowser(**BROWSER_CONFIG)

    WEB_TOOLS = [
        SearchInformationTool(browser),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),
    ]

    text_webbrowser_agent = ToolCallingAgent(
        model=model,
        tools=WEB_TOOLS,
        max_steps=20,
        verbosity_level=2,
        planning_interval=4,
        name="search_agent",
        description=(
            "A team member that will search the internet to answer your question. "
            "Ask him for all your questions that require browsing the web. "
            "Provide him as much context as possible, in particular if you need to search on a specific timeframe! "
            "And don't hesitate to provide him with a complex search task, like finding a difference between two webpages. "
            "Your request must be a real sentence, not a google search!"
        ),
        provide_run_summary=True,
    )
    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += (
        "You can navigate to .txt online files. "
        "If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it. "
        "Additionally, if after some searching you find out that you need more information to answer the question, "
        "you can use `final_answer` with your request for clarification as argument to request for more information."
    )

    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, document_inspection_tool],
        max_steps=12,
        verbosity_level=2,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        planning_interval=4,
        managed_agents=[text_webbrowser_agent],
    )

    answer = manager_agent.run(question)
    return answer, manager_agent
    # translated_answer = translate_to_japanese(answer, model)
    # return translated_answer


def main():
    st.title("Open Deep Research Chat")
    
    model_id = st.text_input("モデルID", value="o1")

    # セッションにチャットの履歴を保持
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    # チャット入力
    user_input = st.chat_input("あなたの質問を入力してください")
    if user_input:
        # ユーザー発言を表示および保存
        st.chat_message("user").markdown(user_input)
        st.session_state.conversation.append({"role": "user", "content": user_input})
        with st.spinner("実行中..."):
            result, manager_agent = run_open_deep_research(user_input, model_id)
        # アシスタントの回答を Markdown で表示・保存
        st.chat_message("assistant").markdown(result)
        st.session_state.conversation.append({"role": "assistant", "content": result})
        for step in manager_agent.memory.steps:
            st.write(step)
    # 過去のチャット履歴がある場合、下部に表示
    if st.session_state.conversation:
        st.markdown("### チャット履歴")
        for entry in st.session_state.conversation:
            role = entry["role"]
            content = entry["content"]
            if role == "user":
                st.markdown(f"**あなた:** {content}")
            else:
                st.markdown("**アシスタント:**")
                st.markdown(content)

if __name__ == "__main__":
    main()