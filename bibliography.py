"""Object bibliography from SIMBAD associations and NASA ADS."""
from astroquery.nasa_ads import ADS
from service_mirrors import configured_simbad


def simbad_papers(main_id):
    client = configured_simbad()
    name = main_id.replace("'", "''")
    papers = []
    cursor = -1
    while True:
        rows = client.query_tap(f"""
            SELECT TOP 500 r.oidbib, r.bibcode, r.title
            FROM basic AS b JOIN has_ref AS h ON h.oidref = b.oid
            JOIN ref AS r ON r.oidbib = h.oidbibref
            WHERE b.main_id = '{name}' AND r.oidbib > {cursor}
            ORDER BY oidbib
        """, maxrec=500)
        papers.extend((str(row["bibcode"]), str(row["title"]), "") for row in rows)
        if len(rows) < 500:
            return papers
        cursor = int(rows[-1]["oidbib"])


def ads_authors_for_simbad(simbad_future):
    """Fetch first authors from ADS for SIMBAD bibcodes absent from ADS name search."""
    records = simbad_future.result()
    bibcodes = list(dict.fromkeys(
        str(record[0]).strip() for record in records if record and record[0]
    ))
    if not bibcodes:
        return []

    client = ADS()
    client.ADS_FIELDS = ["bibcode", "first_author", "author"]
    client.NROWS = 200
    authors = []
    for offset in range(0, len(bibcodes), 40):
        batch = bibcodes[offset:offset + 40]
        terms = " OR ".join(f'bibcode:"{code.replace(chr(34), chr(92) + chr(34))}"'
                            for code in batch)
        response = client.query_simple(
            "(" + terms + ")", get_raw_response=True, cache=False
        )
        for doc in response.json()["response"]["docs"]:
            author_list = doc.get("author") or []
            if isinstance(author_list, str):
                author_list = [author_list]
            first_author = doc.get("first_author") or (author_list[0] if author_list else "")
            if first_author:
                authors.append((doc["bibcode"], "", first_author))
    return authors


def ads_papers(main_id, query_name):
    client = ADS()
    client.ADS_FIELDS = ["bibcode", "title", "first_author", "author"]
    client.NROWS = 200
    client.NSTART = 0
    client.SORT = "bibcode asc"
    names = dict.fromkeys(filter(None, (main_id, query_name)))
    terms = []
    for name in names:
        phrase = name.replace(chr(92), chr(92) * 2).replace('"', chr(92) + '"')
        terms.append(f'full:"{phrase}"')
    query = "collection:astronomy AND (" + " OR ".join(terms) + ")"
    papers = []
    while True:
        response = client.query_simple(query, get_raw_response=True, cache=False)
        page = response.json()["response"]
        for doc in page["docs"]:
            authors = doc.get("author") or []
            if isinstance(authors, str):
                authors = [authors]
            first_author = doc.get("first_author") or (authors[0] if authors else "")
            papers.append((
                doc["bibcode"],
                (doc.get("title") or [""])[0],
                first_author,
            ))
        client.NSTART += len(page["docs"])
        if client.NSTART >= page["numFound"]:
            return papers
        if not page["docs"]:
            raise RuntimeError("ADS returned an incomplete result set.")
