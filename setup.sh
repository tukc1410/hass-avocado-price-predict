mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"youremail@domain\"\n\
" > ~/.streamlit/credentials.toml
echo "[theme]
primaryColor="#4bff4b"
backgroundColor="#789df3"
secondaryBackgroundColor="#4b7ef5"
textColor="#121616"
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml