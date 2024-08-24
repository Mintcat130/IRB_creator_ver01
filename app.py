import streamlit as st
import anthropic

def main():
    st.set_page_config(page_title="병리과 IRB 문서 작성기 📝 ver.01 (HJY)")
    st.title("병리과 IRB 문서 작성기 📝 ver.01 (HJY)")

    # 세션 상태 초기화
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ''
    if 'chat_started' not in st.session_state:
        st.session_state.chat_started = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'client' not in st.session_state:
        st.session_state.client = None

    # API 키 입력 섹션과 연구계획서 작성 버튼을 위한 컨테이너
    input_container = st.empty()

    with input_container.container():
        if not st.session_state.chat_started:
            # API 키 입력 섹션
            api_key = st.text_input("Anthropic API 키를 입력하세요:", type="password", value=st.session_state.api_key)
            if st.button("API 키 확인"):
                st.session_state.api_key = api_key
                st.session_state.client = anthropic.Client(api_key=api_key)
                st.success("API 키가 설정되었습니다!")

            # 연구계획서 작성 버튼
            if st.button("연구계획서 작성하기✏️"):
                if st.session_state.api_key:
                    st.session_state.chat_started = True
                    st.session_state.messages = []  # 채팅 시작 시 메시지 초기화
                    input_container.empty()  # API 키 입력 섹션 숨기기
                else:
                    st.error("먼저 API 키를 입력해주세요.")

    # 채팅 인터페이스
    if st.session_state.chat_started:
        chat_interface()


def chat_interface():
    st.subheader("연구계획서 작성 채팅")

    # 채팅 시작 시 지시사항 표시
    if not st.session_state.messages:
        instruction = """
        KBSMC IRB 연구계획서 작성하기를 시작합니다.
        작성은 "(1) 연구과제명" 항목부터 시작해서 "(18) 자료수집항목 (평가 항목)" 까지 순차적으로 진행됩니다.
        """
        st.info(instruction)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("(1) 연구과제명 부터 작성시작"):
                start_writing("(1) 연구과제명")
        with col2:
            if st.button("작성 원하는 항목 선택하기"):
                st.session_state.show_item_selection = True

    if st.session_state.get('show_item_selection', False):
        st.write("작성할 항목을 선택하세요:")
        items = [
            "(1) 연구과제명", "(2) 연구 관련자", "(3) 실시기관", "(4) 의뢰자(CRO)기관", "(5) 연구 목적",
            "(6) 연구 배경", "(7) 연구 방법", "(8) 자료 수집 및 피싱별 조치", "(9) 선정기준", "(10) 제외기준",
            "(11) 대상자 수 및 산출 근거", "(12) 자료분석과 통계적 방법", "(13) 연구에 활용되는 자료의 기간",
            "(14) 연구예정기간", "(15) 자료 보관 기간 및 폐기 방법", "(16) 연구결과 보고와 출판 방법",
            "(17) 참고 문헌", "(18) 자료 수집 항목 (평가 항목)"
        ]
        
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

        # 버튼을 6열로 배치
        cols = st.columns(6)
        for i, item in enumerate(items):
            with cols[i % 6]:
                if st.button(item, key=item):
                    start_writing(item)


    # 채팅 메시지 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력
    if prompt := st.chat_input("메시지를 입력하세요."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AI 응답 생성
        with st.spinner('AI가 응답을 생성 중입니다...'):
            response = st.session_state.client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1000,
                messages=[
                    {"role": "system", "content": "당신은 병리과 연구자들을 위한 IRB 문서 작성을 돕는 AI 어시스턴트입니다."},
                    *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                ]
            )
            ai_response = response.content[0].text

        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        with st.chat_message("assistant"):
            st.markdown(ai_response)

if __name__ == "__main__":
    main()

def start_writing(item):
    st.session_state.messages.append({
        "role": "assistant", 
        "content": f"{item} 항목에 대한 작성을 시작하겠습니다. 어떤 내용을 작성하시겠습니까?"
    })
    st.session_state.show_item_selection = False  # 항목 선택 UI 숨기기
