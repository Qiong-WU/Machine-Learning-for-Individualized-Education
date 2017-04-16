from __future__ import unicode_literals
from __future__ import print_function
import os
try:
    import youtube_dl
except:
    os.system("pip install youtube-dl")
    print("installing the script, after install,it will exit automaticlly,just restart it.")
try:
    from selenium import webdriver
except:
    os.system("pip install selenium")
    print("installing the script, after install,it will exit automaticlly,just restart it.")
    os._exit(1)

s = raw_input("Please input your search key: ")
l = raw_input("How many videos you want to download: ")
pj = webdriver.PhantomJS(executable_path='./kerwin_browser',service_args=['--load-images=false'])
pj.get('https://youtube.com')
search=pj.find_element_by_css_selector("#masthead-search-term")
search.send_keys(s)
ok=pj.find_element_by_css_selector('#search-btn')
ok.click()
urls = []


if not os.path.exists("./videos"):
    os.mkdir("videos")
    os.chdir('./videos')
else:
    os.chdir('./videos')
while len(urls) <= int(l):  
    current_page = pj.find_elements_by_css_selector('a[href^="/watch"]')
    for i in current_page:
        urls.append(i.get_attribute("href"))
    pj.find_element_by_css_selector("#content > div > div > div > div.branded-page-v2-primary-col > div > div.branded-page-v2-body.branded-page-v2-primary-column-content > div.branded-page-box.search-pager.spf-link > a:nth-child(8)")
ydl_opts = {'proxy':"socks5://127.0.0.1:1080"}
# ydl_opts = {}
for i in urls:
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([i])
        # except youtube_dl.utils.DownloadError:
        except:
            continue


