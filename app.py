import streamlit as st
import anthropic  # Anthropic API 추가

# Anthropic API 클라이언트 초기화 함수
def initialize_anthropic_client(api_key):
    return anthropic.Client(api_key=api_key)

def reset_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.clear()

def start_writing(item):
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": f"{item} 항목에 대한 작성을 시작하겠습니다. 어떤 내용을 작성하시겠습니까?"
    })
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

def generate_ai_response(prompt):
    if 'anthropic_client' in st.session_state:
        response = st.session_state.anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
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

        response = generate_ai_response(prompt)  # AI 응답 생성
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

    if prompt := st.chat_input("메시지를 입력하세요."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = f"AI 응답: {prompt}에 대한 답변입니다."
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

def chat_interface():
    st.subheader("연구계획서 작성 채팅")

    if 'api_key' not in st.session_state or not st.session_state.api_key:
        api_key = st.text_input("Anthropic API 키를 입력하세요:", type="password")
        if st.button("API 키 확인"):
            st.session_state.api_key = api_key
            st.session_state.anthropic_client = initialize_anthropic_client(api_key)  # API 클라이언트 초기화
            st.success("API 키가 설정되었습니다!")
            st.rerun()
        
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
                    st.session_state.chat_started = True
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
