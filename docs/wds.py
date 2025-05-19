'''Advanced web-development server'''

from __future__ import print_function
from bs4 import BeautifulSoup
from collections import defaultdict
import re
import os, glob
import six
if six.PY3:
    from http.server import SimpleHTTPRequestHandler, test


def generate_contents(article_file, output_file):
    # Read the article.html file
    with open(article_file, 'r', encoding='utf-8') as file:
        article_html = file.read()

    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(article_html, 'html.parser')

    # Find all markers and their corresponding headings
    markers = soup.find_all('a', class_='marker')
    toc_items = []

    for marker in markers:
        href = marker.get('href')  # Extract the href attribute
        section_id = marker.get('id')  # Extract the id attribute

        # Find the heading directly following the marker
        next_sibling = marker.find_next_sibling()
        while next_sibling and not next_sibling.name.startswith('h'):
            next_sibling = next_sibling.find_next_sibling()

        # Use the heading text or default to the id if no heading is found
        display_name = next_sibling.get_text(strip=True) if next_sibling else section_id

        if href and section_id and display_name:
            toc_items.append((href, display_name))

    # Organize items into a hierarchical structure based on href
    toc_tree = defaultdict(list)

    def get_parent_id(href):
        """
        Determines the parent section ID based on href. For example:
        - '#section-2.1' -> 'section-2'
        - '#section-3.2' -> 'section-3'
        - Top-level sections will have no parent.
        """
        parts = href.lstrip('#').split('.')
        return '.'.join(parts[:-1]) if len(parts) > 1 else 'top'

    for href, display_name in toc_items:
        parent_id = get_parent_id(href)
        toc_tree[parent_id].append((href, display_name))

    # Recursive function to build nested HTML
    def build_toc_html(parent_id):
        html = ''
        for href, display_name in toc_tree[parent_id]:
            html += f'    <div><a href="{href}">{display_name}</a></div>\n'
            if href.lstrip('#') in toc_tree:  # Check if there are child sections
                html += '      <ul>\n'
                html += build_toc_html(href.lstrip('#'))  # Recursive call for child sections
                html += '      </ul>\n'
        return html

    # Generate the full TOC HTML
    toc_html = '<d-contents class="sticky">\n'
    toc_html += '  <nav class="l-text toc figcaption">\n'
    toc_html += '    <h3>Contents</h3>\n'
    toc_html += build_toc_html('top')  # Start from the top level
    toc_html += '  </nav>\n'
    toc_html += '</d-contents>\n'

    # Write the TOC to contents.html
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(toc_html)

def write_file(fname, fout):
    for s in open(fname):
        if s.startswith('%% '):
            print(s)
            fn = s.split()[1]
            write_file(fn, fout)
        else:
            fout.write(s)

def build():

    with open('index.html', 'w') as fout:
      write_file('main.html', fout)
    print('build finished')


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ['/', '/index.html']:
            build()
        if six.PY3:
            super().do_GET()
        else:
            SimpleHTTPRequestHandler.do_GET(self)

if __name__ == '__main__':
    generate_contents('article.html', 'contents.html')
    build()
    test(HandlerClass=Handler)