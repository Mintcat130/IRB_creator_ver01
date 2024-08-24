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

    # API 키 입력 섹션
    api_key = st.text_input("Anthropic API 키를 입력하세요:", type="password", value=st.session_state.api_key)
    if st.button("API 키 확인"):
        st.session_state.api_key = api_key
        st.success("API 키가 설정되었습니다!")

    # 연구계획서 작성 버튼
    if st.button("연구계획서 작성하기✏️"):
        if st.session_state.api_key:
            st.session_state.chat_started = True
            st.session_state.messages = []  # 채팅 시작 시 메시지 초기화
        else:
            st.error("먼저 API 키를 입력해주세요.")

    # 채팅 인터페이스
    if st.session_state.chat_started:
        chat_interface()

def chat_interface():
    st.subheader("연구계획서 작성 채팅")

    # 채팅 메시지 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력
    if prompt := st.chat_input("메시지를 입력하세요."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AI 응답 (여기서는 간단한 예시 응답을 사용)
        response = f"AI 응답: {prompt}에 대한 답변입니다."
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()
