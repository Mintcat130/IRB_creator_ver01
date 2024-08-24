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

    # API 키 입력 섹션
    api_key = st.text_input("Anthropic API 키를 입력하세요:", type="password", value=st.session_state.api_key)
    if st.button("API 키 확인"):
        st.session_state.api_key = api_key
        st.success("API 키가 설정되었습니다!")

    # 연구계획서 작성 버튼
    if st.button("연구계획서 작성하기✏️"):
        if st.session_state.api_key:
            st.session_state.chat_started = True
        else:
            st.error("먼저 API 키를 입력해주세요.")

    # 채팅 인터페이스
    if st.session_state.chat_started:
        chat_interface()

def chat_interface():
    st.subheader("연구계획서 작성 채팅")
    st.write("여기에 채팅 인터페이스가 구현될 예정입니다.")
    # 여기에 채팅 기능을 추가할 예정입니다.

if __name__ == "__main__":
    main()