import streamlit as st
import anthropic  # Anthropic API 추가

# 시스템 프롬프트
SYSTEM_PROMPT = """
당신은 병리학 분야의 연구 전문가로서 행동하는 AI 조수입니다. 당신의 역할은 사용자가 연구계획서를 작성하는 데 도움을 주는 것입니다. 사용자는 연구계획서의 특정 항목에 대한 정보를 제공할 것이며, 당신은 이를 바탕으로 해당 항목을 작성해야 합니다.

사용자가 제공한 정보를 주의 깊게 분석하고, 당신의 병리학 연구에 대한 전문 지식을 활용하여 요청된 연구계획서 섹션을 작성하세요. 다음 지침을 따르세요:

1. 사용자가 제공한 정보를 최대한 활용하세요.
2. 필요한 경우, 병리학 연구에 대한 당신의 지식을 사용하여 정보를 보완하세요.
3. 연구계획서 섹션의 구조와 형식을 적절히 유지하세요.
4. 명확하고 전문적인 언어를 사용하세요.
5. 필요한 경우 적절한 참고문헌이나 인용을 포함하세요.

한국어로 작성하되 의학 용어는 괄호 안에 영어 원문을 포함시키세요. 예를 들어, "엽상종양(Phyllodes tumor)"과 같은 형식으로 작성하세요.
"""

# 사전 정의된 프롬프트
PREDEFINED_PROMPTS = {
    "연구 배경": "유방의 엽상종양에 대한 연구 배경을 작성해주세요. 발생 빈도, 임상적 중요성, 현재까지의 연구 현황 등을 포함해주세요.",
    "연구 목적": "유방의 엽상종양의 분자유전학적 특성을 분석하여 예후 예측 모델을 개발하는 연구의 목적을 작성해주세요.",
    "연구 방법": "유방의 엽상종양 환자의 조직 샘플을 이용한 유전체 분석과 임상 데이터 분석 방법을 설명해주세요.",
    "기대 효과": "이 연구를 통해 얻을 수 있는 기대 효과와 임상적 의의를 서술해주세요."
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
    
    st.session_state.current_item = item
    st.session_state.chat_started = True
    st.session_state.show_item_selection = False

def show_item_selection():
    st.write("작성할 항목을 선택하세요:")
    items = list(SYSTEM_PROMPTS['prompts'].keys())
    
    cols = st.columns(3)
    for i, item in enumerate(items):
        with cols[i % 3]:
            button_color = "primary" if item == st.session_state.get('current_item', '') else "secondary"
            if item in st.session_state.get('completed_items', []):
                button_label = f"✅ {item}"
            else:
                button_label = item
            if st.button(button_label, key=f"item_{i}", type=button_color):
                start_writing(item)
def generate_ai_response(prompt, current_item):
    if 'anthropic_client' in st.session_state and st.session_state.anthropic_client:
        try:
            system_prompt = f"{SYSTEM_PROMPTS['system_role']}\n\n{SYSTEM_PROMPTS['scope_of_work']}\n\n추가 지시사항: 답변을 작성할 때 번호나 불렛 포인트를 사용하지 말고, 서술형으로 작성해주세요. 문단을 나누어 가독성 있게 작성하되, 전체적으로 하나의 연결된 글이 되도록 해주세요."
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
    current_item = st.session_state.get('current_item', '')
    if current_item:
        st.markdown(f"**현재 작성 중인 항목: {current_item}**")
        instruction = SYSTEM_PROMPTS['prompts'].get(current_item, "이 항목에 대해 어떤 내용을 작성하고 싶으신가요?")
        st.info(instruction)

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
        
        # 항목 완료 처리
        if current_item not in st.session_state.get('completed_items', []):
            st.session_state.completed_items = st.session_state.get('completed_items', []) + [current_item]
        
        # 다음 항목으로 자동 이동
        items = list(SYSTEM_PROMPTS['prompts'].keys())
        current_index = items.index(current_item)
        if current_index < len(items) - 1:
            next_item = items[current_index + 1]
            st.session_state.current_item = next_item
            st.info(f"다음 항목 '{next_item}'으로 이동합니다.")
        else:
            st.success("모든 항목 작성이 완료되었습니다.")
        
        st.rerun()


def chat_interface():
    st.subheader("연구계획서 작성 채팅")

    if 'api_key' not in st.session_state or not st.session_state.api_key:
        api_key = st.text_input("Anthropic API 키를 입력하세요:", type="password")
        
        # API 키 확인 버튼
        if st.button("API 키 확인"):
            client = initialize_anthropic_client(api_key)
            if client:
                st.session_state.api_key = api_key
                st.session_state.anthropic_client = client
                st.success("유효한 API 키입니다. 연구계획서 작성을 시작할 수 있습니다.")
                st.rerun()
            else:
                st.error("API 키 설정에 실패했습니다. 키를 다시 확인해 주세요.")
        
        # 연구계획서 작성하기 버튼 (새로운 줄에 배치)
        if st.button("연구계획서 작성하기✏️"):
            if api_key:
                client = initialize_anthropic_client(api_key)
                if client:
                    st.session_state.api_key = api_key
                    st.session_state.anthropic_client = client
                    st.success("API 키가 설정되었습니다!")
                    st.rerun()
                else:
                    st.error("API 키 설정에 실패했습니다. 키를 다시 확인해 주세요.")
            else:
                st.warning("API 키를 먼저 입력해주세요.")
    else:
        # 사이드바에 홈으로 버튼만 남김
        if st.sidebar.button("🏠홈으로"):
            reset_session()
            st.rerun()

        # 메시지 초기화
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        # 프롬프트 선택 버튼 추가
        cols = st.columns(len(PREDEFINED_PROMPTS))
        for i, (section, prompt) in enumerate(PREDEFINED_PROMPTS.items()):
            if cols[i].button(section):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()

        # 채팅 인터페이스 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("메시지를 입력하세요."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for response in generate_ai_response(prompt):
                    full_response += response
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

    # CSS 스타일 (이전과 동일)
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
