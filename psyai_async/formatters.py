from bs4 import BeautifulSoup
import re
def parse_bluelight_search(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    results = []

    for item in soup.find_all(
        "li", class_="block-row block-row--separated js-inlineModContainer"
    ):
        title_elem = item.find("h3", class_="contentRow-title")
        if title_elem is None:
            title=""
        else:
            title = title_elem.text.strip("\n")
        link = title_elem.find("a")["href"]

        author = item.find("a", class_="username")
        
        if author is None:
            author = "Unknown"
        else:
            author = author.text.strip()
        

        date = item.find("time")["title"]
        date = re.sub(r" at .*", "", date)  # Remove time from date

        forum = item.find_all("li")[-1].text.strip()

        results.append(
            {
                "title": title,
                "link": link,
                "author": author,
                "date": date,
                "forum": forum,
            }
        )

    return results


def create_markdown_list(results):
    markdown = ""
    for i, result in enumerate(results, 1):
        markdown += f"{i}. [{result['title']}](https://www.bluelight.org{result['link']}) - {result['author']}, {result['date']} "
        markdown += f"({result['forum']})\n\n"
    return markdown