import streamlit as st
import anthropic  # Anthropic API 추가

# 시스템 프롬프트 딕셔너리
SYSTEM_PROMPTS = {
    "system_role": "이 LLM 모델은 병리학 분야의 연구 전문가로서 행동하며, 연구계획서 작성의 모든 단계에서 사용자에게 도움을 줍니다. 사용자는 연구계획서의 특정 항목에 대해 필요한 정보를 제공하며, 모델은 이를 기반으로 해당 항목을 작성합니다. 모델은 사용자가 제공하는 정보와 병리학 연구에 대한 지식을 결합하여 연구계획서를 작성합니다.",
    
    "scope_of_work": "사용자는 연구 주제에 대해 자유롭게 기술합니다. 모델은 이 정보를 바탕으로 연구계획서의 '2. 연구 목적' 항목을 먼저 작성합니다. 이후 나머지 항목들이 작성되며, 마지막으로 연구과제명이 결정됩니다. 최종적으로 연구계획서 문서에 해당 내용이 포함됩니다.",
    
    "usage_example": (
        "1. 사용자는 연구 주제에 대해 자유롭게 기술합니다.\n"
        "2. 모델은 사용자의 입력을 바탕으로 줄글 형식으로 '2. 연구 목적'을 작성합니다.\n"
        "3. 연구 목적이 작성된 후, 연구 배경, 연구 방법, 선정 기준, 제외 기준, 대상자 수 및 산출 근거, 자료 분석과 통계적 방법에 대해 차례대로 작성합니다.\n"
        "4. 마지막으로 연구의 핵심 내용을 반영하여 연구과제명을 생성합니다.\n"
        "5. 최종적으로 연구계획서 문서로 출력됩니다."
    ),
    
    "prompts": {
        "user_input": """연구 주제나 키워드에 대해 자유롭게 기술해주세요. 
예시)
- 이 연구를 통해 무엇을 알아내고자 하십니까?
- 어떤 문제를 해결하거나 어떤 가설을 검증하고자 하십니까?
- 이 연구가 왜 중요하다고 생각하십니까?
- 이 연구의 키워드들은 무엇입니까?""",
        
        "2. 연구 목적": "사용자가 연구 주제에 대해 기술한 내용을 바탕으로, 연구의 목적을 번호나 목록 없이 자연스럽게 이어지는 줄글 형식으로 작성하십시오. 연구가 해결하고자 하는 문제, 연구의 중요성, 예상되는 결과 등을 포함하여 설명하십시오.",
        
        "3. 연구 배경": "사용자가 제공한 연구 배경 자료를 바탕으로, 연구의 정당성을 자연스럽게 이어지는 서술형 문장으로 설명하세요. 이론적 배경, 근거, 선행 연구 등을 줄글로 서술하고, 국내외 연구 현황도 반영하세요.",
        
        "4. 연구 방법": "연구 방법을 번호나 목록 없이 서술형 문장으로 자연스럽게 설명하세요. 연구 절차와 방법론을 병리학적 연구에 적합한 방식으로 자세히 기술하세요.",
        
        "5. 선정 기준": "연구 대상자 선정 기준을 자연스럽게 이어지는 줄글 형식으로 명확히 설명하세요. 연구의 목표에 부합하는 대상자 조건을 서술하세요.",
        
        "6. 제외 기준": "연구 대상에서 제외될 기준을 자연스럽게 이어지는 서술형 문장으로 명확히 기술하세요. 연구의 신뢰성을 유지하기 위한 제외 조건을 설명하세요.",
        
        "7. 대상자 수 및 산출 근거": "예상 연구 대상자의 수와 그 산출 근거를 서술형 문장으로 자연스럽게 설명하세요. 필요 시 선행 연구의 통계학적 방법을 참고하여 기술하세요.",
        
        "8. 자료 분석과 통계적 방법": "수집된 자료를 분석하는 방법과 사용할 통계적 방법을 번호나 목록 없이 서술형 문장으로 자연스럽게 설명하세요. 분석 계획과 혼란변수 통제 방법 등을 명확히 기술하세요.",
        
        "9. 연구과제명": "사용자가 앞서 제공한 연구의 목적, 배경, 방법 등을 바탕으로 연구과제명을 줄글 형식으로 자연스럽게 작성하세요. 연구의 핵심을 명확히 반영하는 국문과 영문 제목을 생성하십시오."
    }
}

# Anthropic API 클라이언트 초기화 함수
def initialize_anthropic_client(api_key):
    try:
        client = anthropic.Client(api_key=api_key)
        # 간단한 API 호출로 키 유효성 검사
        client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            messages=[{"role": "user", "content": "Hello"}]
        )
        return client
    except Exception as e:
        st.error(f"API 키 초기화 중 오류 발생: {str(e)}")
        return None

def reset_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.clear()

def start_writing(item):
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'completed_items' not in st.session_state:
        st.session_state.completed_items = []
    
    instruction = SYSTEM_PROMPTS['prompts'].get(item, "이 항목에 대해 어떤 내용을 작성하고 싶으신가요?")
    st.markdown(f"**{item}**")
    st.markdown(instruction)
    
    st.session_state.current_item = item
    st.session_state.chat_started = True
    st.session_state.show_item_selection = False

def show_item_selection():
    st.write("작성할 항목을 선택하세요:")
    items = list(SYSTEM_PROMPTS['prompts'].keys())
    
    cols = st.columns(3)
    for i, item in enumerate(items):
        with cols[i % 3]:
            if st.button(item, key=f"item_{i}"):
                start_writing(item)

def generate_ai_response(prompt, current_item):
    if 'anthropic_client' in st.session_state and st.session_state.anthropic_client:
        try:
            system_prompt = f"{SYSTEM_PROMPTS['system_role']}\n\n{SYSTEM_PROMPTS['scope_of_work']}"
            item_prompt = SYSTEM_PROMPTS['prompts'].get(current_item, "이 항목에 대해 작성해주세요.")
            
            response = st.session_state.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": f"{item_prompt}\n\n사용자 입력: {prompt}"}
                ]
            )
            return response.content[0].text
        except anthropic.APIError as e:
            st.error(f"Anthropic API 오류: {str(e)}")
            return f"AI 응답 생성 중 API 오류가 발생했습니다: {str(e)}"
        except Exception as e:
            st.error(f"예상치 못한 오류 발생: {str(e)}")
            return f"AI 응답을 생성하는 중 예상치 못한 오류가 발생했습니다: {str(e)}"
    else:
        return "API 클라이언트가 초기화되지 않았습니다. API 키를 다시 확인해주세요."


def show_chat_interface():
    # 현재 항목에 대한 지시사항 표시
    current_item = st.session_state.get('current_item', '')
    if current_item:
        instruction = SYSTEM_PROMPTS.get(current_item, "이 항목에 대해 어떤 내용을 작성하고 싶으신가요?")
        st.markdown(f"**{current_item}**")
        st.markdown(instruction)

    for message in st.session_state.get('messages', []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("메시지를 입력하세요."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = generate_ai_response(prompt, current_item)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # 새 메시지가 추가되었을 때만 화면을 갱신합니다.
        st.rerun()


def chat_interface():
    st.subheader("연구계획서 작성 채팅")

    if 'api_key' not in st.session_state or not st.session_state.api_key:
        api_key = st.text_input("Anthropic API 키를 입력하세요:", type="password")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("API 키 확인"):
                client = initialize_anthropic_client(api_key)
                if client:
                    st.session_state.api_key = api_key
                    st.session_state.anthropic_client = client
                    st.success("API 키가 설정되었습니다!")
                    st.rerun()
                else:
                    st.error("API 키 설정에 실패했습니다. 키를 다시 확인해 주세요.")
        with col2:
            if st.button("연구계획서 작성하기✏️"):
                if api_key:
                    st.session_state.api_key = api_key
                    st.session_state.anthropic_client = initialize_anthropic_client(api_key)
                    st.success("API 키가 설정되었습니다!")
                    st.rerun()
                else:
                    st.warning("API 키를 먼저 입력해주세요.")
    else:
        st.sidebar.text(f"현재 API 키: {st.session_state.api_key[:5]}...")
        
        if st.sidebar.button("🏠홈으로"):
            reset_session()
            st.rerun()

        if st.sidebar.button("작성 원하는 항목 선택하기"):
            st.session_state.show_item_selection = True

        if not st.session_state.get('chat_started', False):
            st.info(SYSTEM_PROMPTS['usage_example'])
            
            if st.button("연구 주제 기술하기", key="start_writing"):
                start_writing("1. 연구 주제")

        if st.session_state.get('show_item_selection', False):
            show_item_selection()

        if st.session_state.get('chat_started', False):
            show_chat_interface()
    #Css style

    st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        height: 60px;
        white-space: normal;
        word-wrap: break-word;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    chat_interface()
