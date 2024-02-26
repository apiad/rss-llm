from itertools import groupby
import os
import sys
import random
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import bs4
import dotenv
import requests
import argparse
import datetime


def dateformat(s):
    return datetime.datetime.strptime(s, r"%Y-%m-%dT%H:%M:%S")


parser = argparse.ArgumentParser("python reader.py")
parser.add_argument("--since", type=dateformat, default="2000-01-01T00:00:00")
parser.add_argument("--context", type=int, default=2048)
parser.add_argument("--picks", type=int, default=10)
args = parser.parse_args()

dotenv.load_dotenv()

CLIENT = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))
TOTAL_COST = 0

SUMMARIZER = """
Here is an excerpt from an article:

---
Title: {title}
Author: {author}

Content:
{body}
---

Summarize this article in five sentences or less.
Begin with "{author}...".
"""


def summarize(article):
    global TOTAL_COST

    prompt = SUMMARIZER.format(**article)
    response = CLIENT.chat(
        "mistral-small", messages=[ChatMessage(role="user", content=prompt)]
    )

    TOTAL_COST += response.usage.total_tokens
    return response.choices[0].message.content


ONE_LINER = """
Here is an excerpt from an article:

---
Title: {title}
Author: {author}

Content:
{body}
---

Summarize this article in a single, short sentence.
Begin with "In \"{title}\", {author}...".
"""


def one_liner(article):
    global TOTAL_COST

    prompt = ONE_LINER.format(**article)
    response = CLIENT.chat(
        "mistral-small", messages=[ChatMessage(role="user", content=prompt)]
    )

    TOTAL_COST += response.usage.total_tokens
    return response.choices[0].message.content



CLASSIFIER = """
Here is an excerpt from an article:

---
Title: {title}

Content:
{body}
---

Classify this article in one of the following categories:

- Promotions: if the article is about promoting some product or service.
- News: if the article is discussing recent news or is a digest or review of other articles.
- Tutorials: if the article is a tutorial or lesson aimed to teach the reader some ability.
- Essays: if the article is a personal argumentation on some topic.

Reply only with the category name.
"""


def classify(article):
    global TOTAL_COST

    prompt = CLASSIFIER.format(**article)
    response = CLIENT.chat(
        "mistral-small", messages=[ChatMessage(role="user", content=prompt)]
    )

    TOTAL_COST += response.usage.total_tokens
    return response.choices[0].message.content


with open("feeds.txt") as fp:
    feeds = [l.strip() for l in fp.readlines()]

summaries = []
articles = []
last = args.since

for feed in feeds:
    print(f"\n===\nProcessing: {feed}", file=sys.stderr)

    content = requests.get(feed).text
    tree = bs4.BeautifulSoup(content, "xml")

    for item in tree.find_all("item", recursive=True):
        title = item.title.text
        subtitle = item.subtitle.text if item.subtitle else ""
        author = item.find("dc:creator")
        url = item.link.text

        if author:
            author = author.text
        else:
            author = "the authors"

        content = item.find("content:encoded")
        date = datetime.datetime.strptime(
            str(item.pubDate.string), r"%a, %d %b %Y %H:%M:%S GMT"
        )

        if not content:
            continue

        if date < args.since:
            continue

        last = max(date, last)

        body = bs4.BeautifulSoup(content.text, "lxml").get_text("\n")

        article = dict(
            title=title,
            subtitle=subtitle,
            author=author,
            words=len(body.split()),
            body=body[: args.context],
            date=date,
            url=url,
        )

        print(title, file=sys.stderr)
        articles.append(article)


for article in articles:
    article["category"] = classify(article)
    article["short"] = one_liner(article)

    print("", file=sys.stderr)
    print(article['title'], file=sys.stderr)
    print(article['category'], file=sys.stderr)
    print(article['short'], file=sys.stderr)


editorial_picks = []
selected_authors = set()
selected_titles = set()

random.shuffle(articles)

for article in articles:
    if article['category'] in ["Promotions"]:
        continue

    if article["words"] < 1000:
        continue

    if article['author'] in selected_authors:
        continue

    editorial_picks.append(article)
    selected_authors.add(article['author'])

editorial_picks = editorial_picks[:args.picks]
selected_titles = [article['title'] for article in editorial_picks]

for article in editorial_picks:
    article["long"] = summarize(article)
    print("", file=sys.stderr)
    print(article['title'], file=sys.stderr)
    print(article['long'], file=sys.stderr)


SUMMARY = """
Here is a collection of one-line summaries of articles
for a newsletter digest.

---
{summaries}
---

Write one editorial paragraph, around five sentences long, explaining in broad terms
what are the main topics discussed in those articles.
Combine several related topics in a single sentence when possible.
Do no mention titles or author names.
Use evocative language and strong verbs to encourage readers
to check the articles in more detail.
End with a one-liner that summarizes what is, in general, this issue about.
"""

prompt = SUMMARY.format(summaries="\n\n".join(article['short'] for article in articles))
response = CLIENT.chat(
    "mistral-small", messages=[ChatMessage(role="user", content=prompt)]
)

TOTAL_COST += response.usage.total_tokens
summary = response.choices[0].message.content

print(f"\nTotal cost: {TOTAL_COST}", file=sys.stderr)

print("# Newsletter Digest\n")

print(f"> Timestamp: {last.strftime(r'%Y-%m-%dT%H:%M:%S')}\n")

print(summary)
print()

print("## Featured articles\n")

for article in editorial_picks:
    print(f"### {article['title']}\n")
    print(article['url'])
    print()
    text: str = article['long'].replace(article['title'], f"[{article['title']}]({article['url']})")
    print(text)
    print()

# articles = [a for a in articles if a['title'] not in selected_titles]
articles.sort(key=lambda a: a['category'])

print("## Other articles\n")

for category, items in groupby(articles, key=lambda x: x['category']):
    print(f"### {category}\n")

    for item in items:
        text: str = item['short'].replace(item['title'], f"[{item['title']}]({item['url']})")

        print(text)
        print()
