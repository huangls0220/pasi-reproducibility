# Release readiness and unresolved author actions

## Completed

- Public repository is anonymously readable.
- Environment lock, runnable commands, source tests, seed-level results, manifests, privacy audit, and legal GeoLife reconstruction instructions are present.
- The manuscript points to an immutable commit, not only to the moving `main` branch.
- Restricted GeoLife inputs are excluded from the repository and public archives.
- Negative QoS, coverage, and public-mechanism comparison results are retained.

## Pending before an archival DOI release

1. All authors must approve a software license. Public visibility alone does not grant reuse rights. No license is added until this approval exists.
2. All author names, ORCIDs if used, manuscript title, venue status, and contact details must be finalized before creating `CITATION.cff` or Zenodo metadata. Only the first author's identity is currently confirmed, so a partial citation record would be misleading.
3. R75 has finished locally. Merge the exact frozen runner and non-sensitive outputs, run the package verifier from a clean clone, and replace the pending row in `ARTIFACT_VERSIONS.md` with the full merge SHA.
4. Enable the repository in Zenodo, create a GitHub release from that verified commit, and copy the DOI returned by Zenodo into the manuscript and repository. Do not reserve or invent a DOI in advance.

## Recommended author decision

If every code contributor agrees, use a standard permissive license such as MIT. This is a legal authorship decision, not a technical default; the repository therefore continues to state “no software license granted” until approval is documented.
