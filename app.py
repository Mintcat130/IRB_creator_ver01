import streamlit as st
import anthropic

def reset_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.clear()  # 추가적인 안전장치로 세션 상태를 완전히 비웁니다.

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
    
    # 버튼을 6열로 배치
    cols = st.columns(6)
    for i, item in enumerate(items):
        with cols[i % 6]:
            if st.button(item, key=item):
                start_writing(item)

def show_chat_interface():
    for message in st.session_state.get('messages', []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력
    if prompt := st.chat_input("메시지를 입력하세요."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 여기에 AI 응답 생성 로직을 추가합니다.
        # 예: response = generate_ai_response(prompt)
        response = f"AI 응답: {prompt}에 대한 답변입니다."
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

def chat_interface():
    st.subheader("연구계획서 작성 채팅")

    # 사이드바에 "🏠홈으로" 버튼 추가
    if st.sidebar.button("🏠홈으로"):
        reset_session()
        st.rerun()  # st.experimental_rerun() 대신 st.rerun() 사용

    # 사이드바에 "작성 원하는 항목 선택하기" 버튼 추가
    if st.sidebar.button("작성 원하는 항목 선택하기"):
        st.session_state.show_item_selection = True

    # API 키 입력 섹션
    if 'api_key' not in st.session_state or not st.session_state.api_key:
        api_key = st.text_input("Anthropic API 키를 입력하세요:", type="password")
        if st.button("API 키 확인"):
            st.session_state.api_key = api_key
            st.success("API 키가 설정되었습니다!")
            st.rerun()  # 여기도 st.rerun() 사용

    # API 키가 설정된 후 채팅 인터페이스 표시
    if 'api_key' in st.session_state and st.session_state.api_key:
        if not st.session_state.get('chat_started', False):
            instruction = """
            KBSMC IRB 연구계획서 작성하기를 시작합니다.
            작성은 "(1) 연구과제명" 항목부터 시작해서 "(18) 자료수집항목 (평가 항목)" 까지 순차적으로 진행됩니다.
            """
            st.info(instruction)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("(1) 연구과제명 부터 작성시작"):
                    st.session_state.chat_started = True
                    start_writing("(1) 연구과제명")
            with col2:
                if st.button("작성 원하는 항목 선택하기"):
                    st.session_state.show_item_selection = True

        if st.session_state.get('show_item_selection', False):
            show_item_selection()

        if st.session_state.get('chat_started', False):
            show_chat_interface()

# CSS를 사용하여 버튼 스타일 지정
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
