import streamlit as st
import anthropic
import PyPDF2
import io
import requests
from scholarly import scholarly
from Bio import Entrez
import json
import re

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

# PREDEFINED_PROMPTS 수정
PREDEFINED_PROMPTS = {
    "2. 연구 목적": """
    사용자가 제공한 연구 주제와 키워드를 바탕으로, 연구 목적과 가설을 1000자 이내의 줄글로 작성하세요. 어미는 반말 문어체로 합니다. (예: ~하였다. ~있다. ~있었다)
    다음 사항을 포함해야 합니다:
    1. 연구의 주요 목적
    2. 연구로 인해 의도하는 가설
    3. 가설을 입증하기 위한 구체적인 설명
    4. 연구의 중요성과 예상되는 결과

    사용자 입력:
    {user_input}

    위의 내용을 바탕으로 연구 목적과 가설을 구체화하여 작성해주세요.
    """,
    "3. 연구 배경": """
    제공된 정보를 바탕으로 연구의 배경을 1500자 이내로 설명해주세요. 어미는 반말 문어체로 합니다. (예: ~하였다. ~있다. ~있었다)
    다음 사항을 포함하세요:
    1. 이론적 배경 및 근거
    2. 선행 연구 및 결과
    3. 연구 배경과 연구의 정당성에 대한 설명
    4. 국내외 연구 현황

    사용자 입력:
    {user_input}

    연구 목적:
    {research_purpose}

    검색된 논문:
    {papers}

    PDF 내용:
    {pdf_content}

    위의 내용을 바탕으로 연구 배경을 구체화하여 작성해주세요. 참고문헌을 인용할 때는 [저자, 연도] 형식으로 표기해주세요.
    """,

    "4. 선정기준": """
    2, 3번 섹션의 결과물과 참고한 논문들을 토대로, 이 연구에 적당한 대상자 그룹을 추천해주세요. 다음 지침을 따라주세요:
    1. 구체적인 년도나 시기는 적지 않습니다. (잘못된 예시: 2009년 국가 건강검진을 받은 4,234,415명)
    2. 어미는 적지 않고 단어로 문장을 끝냅니다.
    3. 선정기준 예시: 40세에서 60세 사이에 해당하며, 이전 치매에 진단받은 과거력이 없는 수검자

    연구 목적:
    {research_purpose}

    연구 배경:
    {research_background}

    위의 내용을 바탕으로 적절한 선정기준을 작성해주세요.
    """
}


# 연구 섹션 순서 정의
RESEARCH_SECTIONS = [
    "2. 연구 목적",
    "3. 연구 배경",
    "4. 선정기준",
    # 다른 섹션들은 나중에 추가할 예정입니다.
]

# Anthropic API 클라이언트 초기화 함수
def initialize_anthropic_client(api_key):
    try:
        client = anthropic.Client(api_key=api_key)
        # 간단한 API 호출로 키 유효성 검사
        client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=2000,
            messages=[{"role": "user", "content": "Hello"}]
        )
        return client
    except Exception as e:
        st.error(f"API 키 초기화 중 오류 발생: {str(e)}")
        return None

#세션 초기화 함수
def reset_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.clear()

                
#AI 응답 생성 함수
def generate_ai_response(prompt):
    if 'anthropic_client' in st.session_state and st.session_state.anthropic_client:
        try:
            system_prompt = f"{SYSTEM_PROMPT}\n\n추가 지시사항: 답변을 작성할 때 번호나 불렛 포인트를 사용하지 말고, 서술형으로 작성해주세요. 문단을 나누어 가독성 있게 작성하되, 전체적으로 하나의 연결된 글이 되도록 해주세요."
            
            response = st.session_state.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=2000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
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


# PDF 파일 업로드 함수
def upload_pdf():
    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")
    if uploaded_file is not None:
        return extract_text_from_pdf(uploaded_file)
    return None

# PDF에서 텍스트 추출 함수
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.getvalue()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# PubMed 검색 함수 (수정)
def search_pubmed(query, max_results=10):
    Entrez.email = "your_email@example.com"
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    
    ids = record["IdList"]
    results = []
    for id in ids:
        handle = Entrez.efetch(db="pubmed", id=id, rettype="medline", retmode="text")
        record = Entrez.read(Entrez.parse(handle))
        if record:
            article = record[0]
            title = article.get("TI", "No title")
            year = article.get("DP", "")[:4]  # 출판 연도
            authors = ", ".join(article.get("AU", []))[:50] + "..." if len(article.get("AU", [])) > 2 else ", ".join(article.get("AU", []))
            link = f"https://pubmed.ncbi.nlm.nih.gov/{id}/"
            results.append({"title": title, "year": year, "authors": authors, "link": link})
        handle.close()
    return results

# Google Scholar 검색 함수 수정
def search_google_scholar(query, max_results=10):
    search_query = scholarly.search_pubs(query)
    results = []
    for i, result in enumerate(search_query):
        if i >= max_results:
            break
        try:
            title = result['bib'].get('title', 'No title')
            year = result['bib'].get('pub_year', 'No year')
            authors = result['bib'].get('author', 'No author')
            if isinstance(authors, list):
                authors = ", ".join(authors[:2]) + "..." if len(authors) > 2 else ", ".join(authors)
            link = result.get('pub_url', '#')
            results.append({"title": title, "year": year, "authors": authors, "link": link})
        except AttributeError:
            continue  # 결과를 건너뛰고 다음 결과로 진행
    return results


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

def write_research_purpose():
    st.markdown("## 2. 연구 목적")
    st.markdown("어떤 연구를 계획중인지, 연구에 대한 내용이나 키워드를 형식에 상관없이 자유롭게 입력해주세요. 입력 후 버튼을 누르면 AI 모델이 연구목적에 대한 줄글을 작성 해 줍니다.")
    
    user_input = st.text_area("연구 주제 및 키워드:", height=150)
    
    if st.button("연구 목적 생성"):
        if user_input:
            prompt = PREDEFINED_PROMPTS["2. 연구 목적"].format(user_input=user_input)
            ai_response = generate_ai_response(prompt)
            
            st.session_state.section_contents["2. 연구 목적"] = ai_response
            st.markdown("### AI가 생성한 연구 목적:")
            st.markdown(ai_response)
            
            # 글자 수 확인
            char_count = len(ai_response)
            st.info(f"생성된 내용의 글자 수: {char_count}/1000")
            
            if char_count > 1000:
                st.warning("생성된 내용이 1000자를 초과했습니다. 수정이 필요할 수 있습니다.")
        else:
            st.warning("연구 주제나 키워드를 입력해주세요.")

    # 편집 기능
    if "2. 연구 목적" in st.session_state.section_contents:
        edited_content = st.text_area(
            "생성된 내용을 편집하세요:",
            st.session_state.section_contents["2. 연구 목적"],
            height=300
        )
        if st.button("편집 내용 저장"):
            st.session_state.section_contents["2. 연구 목적"] = edited_content
            st.success("편집된 내용이 저장되었습니다.")


# 연구 배경 작성 함수 (수정)
def write_research_background():
    st.markdown("## 3. 연구 배경")
    
    # 키워드 입력
    keywords = st.text_input("연구 배경 작성을 위한 참조논문 검색에 사용할 키워드를 입력하세요 (최대 10개, 쉼표로 구분):")
    keywords_list = [k.strip() for k in keywords.split(',') if k.strip()][:10]
    
    if keywords_list:
        st.write("입력된 키워드:", ", ".join(keywords_list))
        
    if st.button("논문 검색"):
        if keywords_list:
            search_query = " ".join(keywords_list)
            
            with st.spinner("논문을 검색 중입니다..."):
                pubmed_results = search_pubmed(search_query)
                scholar_results = search_google_scholar(search_query)
            
            st.success("검색이 완료되었습니다.")
            
            # PubMed 결과 표시
            st.subheader("PubMed 검색 결과")
            for i, result in enumerate(pubmed_results):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"[{result['title']} ({result['year']})]({result['link']})")
                    st.caption(f"저자: {result['authors']}")
                with col2:
                    if st.button("삭제", key=f"del_pubmed_{i}"):
                        del pubmed_results[i]
                        st.rerun()
            
           # Google Scholar 결과 표시
            st.subheader("Google Scholar 검색 결과")
            for i, result in enumerate(scholar_results):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"[{result['title']} ({result['year']})]({result['link']})")
                    st.caption(f"저자: {result['authors']}")
                with col2:
                    if st.button("삭제", key=f"del_scholar_{i}"):
                        del scholar_results[i]
                        st.rerun()
            
            # 검색 결과 세션 상태에 저장
            st.session_state.pubmed_results = pubmed_results
            st.session_state.scholar_results = scholar_results
        else:
            st.warning("키워드를 입력해주세요.")
            
    # PDF 파일 업로드
    uploaded_files = st.file_uploader("연구 배경 작성에 참고할 PDF 파일을 업로드하세요 (여러 개 선택 가능)", type="pdf", accept_multiple_files=True)
    
    pdf_texts = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            pdf_text = extract_text_from_pdf(uploaded_file)
            pdf_texts.append(pdf_text)
        st.success(f"{len(uploaded_files)}개의 PDF 파일이 성공적으로 업로드되었습니다.")
    
    # 연구 배경 생성 버튼
    if st.button("해당 내용으로 연구배경 작성하기"):
        if 'pubmed_results' in st.session_state or 'scholar_results' in st.session_state or pdf_texts:
            # 연구 목적 가져오기
            research_purpose = st.session_state.section_contents.get("2. 연구 목적", "")
            
            # 검색 결과 및 PDF 내용 결합
            papers = []
            if 'pubmed_results' in st.session_state:
                papers.extend([f"{r['title']} ({r['year']})" for r in st.session_state.pubmed_results])
            if 'scholar_results' in st.session_state:
                papers.extend([f"{r['title']} ({r['year']})" for r in st.session_state.scholar_results])
            papers_text = "\n".join(papers)
            
            pdf_content = "\n".join(pdf_texts)
            
            # AI에 전달할 프롬프트 생성
            prompt = PREDEFINED_PROMPTS["3. 연구 배경"].format(
                user_input=keywords,
                research_purpose=research_purpose,
                papers=papers_text,
                pdf_content=pdf_content[:1000]  # PDF 내용은 1000자로 제한
            )
            
            # AI 응답 생성
            ai_response = generate_ai_response(prompt)
            
            st.session_state.section_contents["3. 연구 배경"] = ai_response
            st.markdown("### AI가 생성한 연구 배경:")
            st.markdown(ai_response)
            
            # 글자 수 확인
            char_count = len(ai_response)
            st.info(f"생성된 내용의 글자 수: {char_count}/1500")
            
            if char_count > 1500:
                st.warning("생성된 내용이 1500자를 초과했습니다. 수정이 필요할 수 있습니다.")
        else:
            st.warning("논문을 검색하거나 PDF를 업로드한 후 다시 시도해주세요.")

    # 편집 기능
    if "3. 연구 배경" in st.session_state.section_contents:
        edited_content = st.text_area(
            "생성된 내용을 편집하세요:",
            st.session_state.section_contents["3. 연구 배경"],
            height=300
        )
        if st.button("편집 내용 저장"):
            st.session_state.section_contents["3. 연구 배경"] = edited_content
            st.success("편집된 내용이 저장되었습니다.")

# 선정기준 작성 함수
def write_selection_criteria():
    st.markdown("## 4. 선정기준")
    
    # 편집 기능 (AI 추천 전에도 표시)
    edited_content = st.text_area(
        "생성된 내용을 편집하거나, 선정기준을 직접 작성하세요:",
        st.session_state.section_contents.get("4. 선정기준", ""),
        height=200
    )
    
    if st.button("선정기준 AI에게 추천받기"):
        research_purpose = st.session_state.section_contents.get("2. 연구 목적", "")
        research_background = st.session_state.section_contents.get("3. 연구 배경", "")
        
        prompt = PREDEFINED_PROMPTS["4. 선정기준"].format(
            research_purpose=research_purpose,
            research_background=research_background
        )
        
        ai_response = generate_ai_response(prompt)
        
        st.session_state.section_contents["4. 선정기준"] = ai_response
        st.markdown("### AI가 추천한 선정기준:")
        st.markdown(ai_response)
    
    if st.button("편집 내용 저장"):
        st.session_state.section_contents["4. 선정기준"] = edited_content
        st.success("편집된 내용이 저장되었습니다.")

def extract_references(text):
    # [저자, 연도] 형식의 참고문헌을 추출
    references = re.findall(r'\[([^\]]+)\]', text)
    return list(set(references))  # 중복 제거

# 여기에 chat_interface 함수가 이어집니다.

def chat_interface():
    st.subheader("연구계획서 작성 채팅")

    if 'api_key' not in st.session_state or not st.session_state.api_key:
        api_key = st.text_input("Anthropic API 키를 입력하세요:", type="password")
        
        # API 키 확인 버튼
        if st.button("API 키 확인"):
            client = initialize_anthropic_client(api_key)
            if client:
                st.success("유효한 API 키입니다. 연구계획서 작성하기 버튼을 눌러 시작하세요.")
                st.session_state.temp_api_key = api_key  # 임시로 API 키 저장
            else:
                st.error("API 키 설정에 실패했습니다. 키를 다시 확인해 주세요.")
        
        # 연구계획서 작성하기 버튼
        if st.button("연구계획서 작성하기✏️"):
            if 'temp_api_key' in st.session_state:
                st.session_state.api_key = st.session_state.temp_api_key
                st.session_state.anthropic_client = initialize_anthropic_client(st.session_state.api_key)
                del st.session_state.temp_api_key  # 임시 저장된 API 키 삭제
                st.success("API 키가 설정되었습니다!")
                st.rerun()
            else:
                st.warning("먼저 API 키를 입력하고 확인해주세요.")
    else:
        # API 키가 이미 설정된 경우의 로직
        if 'current_section' not in st.session_state:
            st.session_state.current_section = RESEARCH_SECTIONS[0]
        if 'section_contents' not in st.session_state:
            st.session_state.section_contents = {}
        if 'references' not in st.session_state:
            st.session_state.references = []  # 참고문헌 리스트 초기화

        if 'api_key' in st.session_state:
            st.sidebar.text(f"현재 API 키: {st.session_state.api_key[:5]}...")
        
    
        
        if st.sidebar.button("🏠홈으로"):
            reset_session()
            st.rerun()

        # 현재 섹션에 따른 작성 인터페이스 표시
        if st.session_state.current_section == "2. 연구 목적":
            write_research_purpose()
        elif st.session_state.current_section == "3. 연구 배경":
            write_research_background()
        elif st.session_state.current_section == "4. 선정기준":
            write_selection_criteria()
             # ... (다른 섹션들에 대한 조건문 추가)

      # 다음 섹션으로 이동
        if st.button("다음 섹션"):
            current_index = RESEARCH_SECTIONS.index(st.session_state.current_section)
            if current_index < len(RESEARCH_SECTIONS) - 1:
                st.session_state.current_section = RESEARCH_SECTIONS[current_index + 1]
                st.rerun()
            else:
                st.success("모든 섹션을 완료했습니다!")

        # 전체 내용 미리보기
        if st.sidebar.button("전체 내용 미리보기"):
            for section in RESEARCH_SECTIONS:
                st.markdown(f"### {section}")
                st.markdown(st.session_state.section_contents.get(section, "아직 작성되지 않았습니다."))
            
            # 참고문헌 표시
            if st.session_state.references:
                st.markdown("### 참고문헌")
                for ref in st.session_state.references:
                    st.markdown(f"- {ref}")

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

