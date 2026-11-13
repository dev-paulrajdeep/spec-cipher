# SpecCipher Corpus and Provenance

This glossary defines the document and page identities used when building a corpus and citing recommendation evidence.

## Publications and sources

**Logical document**:
One complete publication or edition, regardless of how many physical files contain it.
_Avoid_: PDF, volume, source file

**Source file**:
One ingested physical file. Its identity and page offset remain distinct from the logical document it belongs to.
_Avoid_: document, publication

**Missing volume**:
A declared but un-ingested page range within a multipart publication. It records an identity and range without representing its contents as present.
_Avoid_: empty volume, placeholder pages

**Standard stub**:
A standard mentioned by corpus evidence but not ingested as a logical document. It may be shown as a related reference and is never eligible as a primary recommendation.
_Avoid_: indexed standard, candidate document

## Page provenance

**Physical page**:
A one-based page number within one source file.
_Avoid_: absolute page, publication page

**Absolute page**:
A continuous one-based page number within the logical publication, calculated from the source file's declared offset and physical page.
_Avoid_: physical page, printed page

**Printed page label**:
The human-visible page label printed on a page, such as `Page 405` or `Annex A-3`. It may differ from both physical and absolute page numbers.
_Avoid_: page number
