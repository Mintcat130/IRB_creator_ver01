import streamlit as st
import anthropic  # Anthropic API 추가

# 시스템 프롬프트 딕셔너리
SYSTEM_PROMPTS = {
    "(1) 연구과제명": "사용자가 연구 주제에 대해 자유롭게 기술한 내용을 바탕으로, 연구의 목적과 주제를 명확히 나타내는 연구과제명을 작성하세요. 국문과 영문으로 작성하십시오.",
    "(5) 연구 목적": "사용자가 제공한 연구 목적에 대한 설명을 바탕으로, 연구의 가설을 명확히 하고, 이를 입증하기 위한 구체적인 설명을 작성하세요.",
    "(6) 연구 배경": "사용자가 제공한 연구 배경 자료와 관련된 이론적 배경, 근거, 선행 연구 등을 바탕으로 연구의 정당성을 설명하세요. 국내외 연구 현황을 반영하세요.",
    "(7) 연구 방법": "사용자가 제시한 연구 방법에 대한 기본 정보를 토대로, 연구 절차와 방법론을 상세히 설명하세요. 병리학적 연구에 적합한 연구 방법을 제안하고 기술하세요.",
    "(9) 선정기준": "연구 대상자 선정 기준을 명확히 기술하세요. 연구의 목표에 부합하는 대상자 조건을 설정하세요.",
    "(10) 제외기준": "연구 대상에서 제외될 기준을 명확히 기술하세요. 연구의 신뢰성을 유지하기 위한 제외 조건을 설정하세요.",
    "(11) 대상자 수 및 산출 근거": "예상 연구 대상자의 수와 그 산출 근거를 작성하세요. 필요 시 선행 연구의 통계학적 방법을 참고하여 설명하세요.",
    "(12) 자료분석과 통계적 방법": "수집된 자료를 분석하는 방법과 사용할 통계적 방법을 기술하세요. 분석 계획, 혼란변수 통제 방법 등을 명확히 하세요."
}

# Anthropic API 클라이언트 초기화 함수
def initialize_anthropic_client(api_key):
    try:
        client = anthropic.Client(api_key=api_key)
        # 간단한 API 호출로 키 유효성 검사
        client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=10,
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
    
    if item == "(1) 연구과제명":
        instruction = """
        연구 주제나 키워드에 대해 자유롭게 기술해주세요. 
        예시)
           - 이 연구를 통해 무엇을 알아내고자 하십니까?
           - 어떤 문제를 해결하거나 어떤 가설을 검증하고자 하십니까?
           - 이 연구가 왜 중요하다고 생각하십니까?
           - 이 연구의 키워드들은 무엇입니까?
        """
    else:
        instruction = "이 항목에 대해 어떤 내용을 작성하고 싶으신가요?"

    st.session_state.messages.append({
        "role": "assistant", 
        "content": f"{item} 항목에 대한 작성을 시작하겠습니다.\n\n{instruction}"
    })
    st.session_state.current_item = item
    st.session_state.chat_started = True
    st.session_state.show_item_selection = False

def show_item_selection():
    st.write("작성할 항목을 선택하세요:")
    items = [
        "(1) 연구과제명", "(2) 연구 관련자", "(3) 실시기관", "(4) 의뢰자(CRO)기관", "(5) 연구 목적",
        "(6) 연구 배경", "(7) 연구 방법", "(8) 자료 수집 및 피싱별 조치", "(9) 선정기준", "(10) 제외기준",
        "(11) 대상자 수 및 산출 근거", "(12) 자료분석과 통계적 방법", "(13) 연구에 활용되는 자료의 기간",
        "(14) 연구예정기간", "(15) 자료 보관 기간 및 폐기 방법", "(16) 연구결과 보고와 출판 방법",
        "(17) 참고 문헌", "(18) 자료 수집 항목 (평가 항목)"
    ]
    
    cols = st.columns(6)
    for i, item in enumerate(items):
        with cols[i % 6]:
            if st.button(item, key=f"item_{i}"):
                start_writing(item)

def generate_ai_response(prompt, current_item):
    if 'anthropic_client' in st.session_state and st.session_state.anthropic_client:
        try:
            system_prompt = SYSTEM_PROMPTS.get(current_item, "당신은 병리과 연구자들을 위한 IRB 문서 작성을 돕는 AI 어시스턴트입니다.")
            response = st.session_state.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            st.error(f"AI 응답 생성 중 오류 발생: {str(e)}")
            return "AI 응답을 생성하는 중 오류가 발생했습니다. 다시 시도해 주세요."
    else:
        return "API 클라이언트가 초기화되지 않았습니다. API 키를 다시 확인해주세요."


def show_chat_interface():
    for message in st.session_state.get('messages', []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("메시지를 입력하세요."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        current_item = st.session_state.get('current_item', "")
        response = generate_ai_response(prompt, current_item)  # current_item 추가
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # 새 메시지가 추가되었을 때만 화면을 갱신합니다.
        st.rerun()


def chat_interface():
    st.subheader("연구계획서 작성 채팅")

    if 'api_key' not in st.session_state or not st.session_state.api_key:
        api_key = st.text_input("Anthropic API 키를 입력하세요:", type="password")
          if st.button("API 키 확인"):
            client = initialize_anthropic_client(api_key)
            if client:
                st.session_state.api_key = api_key
                st.session_state.anthropic_client = client
                st.success("API 키가 설정되었습니다!")
                st.rerun()
    else:
        st.error("올바르지 않은 API 키입니다. 다시 확인해 주세요.")
        
        if st.button("연구계획서 작성하기✏️"):
            st.warning("API 키를 먼저 입력해주세요.")
    else:
        if st.sidebar.button("🏠홈으로"):
            reset_session()
            st.rerun()

        if st.sidebar.button("작성 원하는 항목 선택하기"):
            st.session_state.show_item_selection = True

        if not st.session_state.get('chat_started', False):
            instruction = """
            KBSMC IRB 연구계획서 작성하기를 시작합니다.
            작성은 "(1) 연구과제명" 항목부터 시작해서 "(18) 자료수집항목 (평가 항목)" 까지 순차적으로 진행됩니다.
            """
            st.info(instruction)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("(1) 연구과제명 부터 작성시작", key="start_writing"):
                    start_writing("(1) 연구과제명")
            with col2:
                if st.button("작성 원하는 항목 선택하기", key="select_item"):
                    st.session_state.show_item_selection = True

        if st.session_state.get('show_item_selection', False):
            show_item_selection()

        if st.session_state.get('chat_started', False):
            show_chat_interface()

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
