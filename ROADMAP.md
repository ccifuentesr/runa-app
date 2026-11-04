# RUNA roadmap

This roadmap describes planned work on RUNA as a scientific software application. Estimates are indicative and are intended to make the scope reviewable; priorities may change as catalogue services and data releases evolve.

The planned work focuses on scientific traceability, catalogue integration, reliability, testing, documentation and user-facing workflows. It does not include training, fine-tuning, benchmarking or evaluating new machine-learning models.

## 1. Object bibliography and provenance (8–12 hours)

- Retrieve the bibliography associated with the selected object from SIMBAD.
- Group references by measurement type: astrometry, photometry, spectroscopy, multiplicity, activity, rotation, planets and fundamental parameters.
- Link each displayed value to its catalogue, publication or archive record when the source provides that information.
- Show the query service, catalogue release, retrieval time and matching method used by each RUNA panel.
- Add a compact provenance export so users can reproduce or audit an object report.

## 2. Bayesian distances from Bailer-Jones et al. (2021) (6–9 hours)

- Query the Gaia EDR3 distance catalogue of Bailer-Jones et al. (2021) when a compatible source identifier is available.
- Display geometric and photogeometric distance estimates with their lower and upper credible bounds.
- Keep inverse-parallax distance visible for comparison and explain when it becomes unreliable.
- Identify Gaia release compatibility explicitly instead of silently mixing DR2, EDR3 and DR3 identifiers.
- Propagate the selected distance estimate consistently to absolute magnitudes and position plots.

## 3. Extended Mamajek reference relations (7–10 hours)

- Expand the current reference view to the useful temperature, colour, bolometric-correction, luminosity, radius, mass and absolute-magnitude relations in the Mamajek dwarf sequence.
- Add interpolation with explicit validity limits and no extrapolation beyond the published sequence.
- Display the upstream table version, citation and provenance beside derived values.
- Separate RUNA's transformation code from the third-party reference data.
- Use the upstream table only under confirmed redistribution terms; otherwise retrieve it from its authoritative source or require a user-supplied copy.

## 4. Gaia DR4 readiness (8–12 hours)

- Introduce a release-aware Gaia data adapter so query and display code do not depend directly on DR3 column names.
- Keep catalogue identifiers, table names and quality cuts in a versioned schema configuration.
- Add compatibility tests using saved, non-sensitive sample responses.
- Document which RUNA products are release-specific and which can migrate automatically.
- Prepare the interface for future DR4 products such as updated astrometry, source classifications, epoch data and radial velocities without presenting unavailable fields as current functionality.

## 5. Reproducible object reports (6–9 hours)

- Export a structured report containing identifiers, measurements, derived parameters, warnings, citations and provenance.
- Provide machine-readable JSON and CSV exports in addition to the current panel-specific downloads.
- Record units, missing-value semantics and catalogue release for every exported field.
- Add a human-readable report suitable for attaching to observing notes or target-selection records.

## 6. Catalogue matching and uncertainty handling (7–11 hours)

- Centralise coordinate propagation and angular matching across catalogue and archive queries.
- Report match radius, epoch and separation instead of presenting a counterpart without context.
- Add warnings for crowded fields, large proper motion, ambiguous counterparts and inconsistent identifiers.
- Standardise uncertainty propagation for distances, absolute magnitudes and empirical relations.
- Distinguish measurements, catalogue-derived quantities and RUNA estimates visually and in exports.

## 7. Remote-service reliability (6–9 hours)

- Apply consistent timeouts, retries and user-facing error categories to SIMBAD, VizieR, Gaia, MAST, ESO and exoplanet services.
- Cache stable catalogue responses while allowing users to request a fresh query.
- Detect rate limiting and temporary service outages without reporting them as missing astronomical data.
- Add service-status details to downloadable provenance.
- Provide graceful degradation when optional services or packages are unavailable.

## 8. Architecture and automated tests (12–18 hours)

- Split the Streamlit interface into smaller views, services and scientific calculation modules.
- Add unit tests for coordinate handling, photometric conversions, empirical relations and export schemas.
- Add mocked integration tests for remote catalogue responses, empty results and malformed values.
- Add regression tests for representative nearby stars without storing personal or confidential data.
- Configure continuous checks for syntax, tests and dependency installation.

## 9. Documentation and data governance (5–8 hours)

- Add a data-source inventory covering bundled tables, remote services, citations, versions and applicable terms.
- Document installation, optional dependencies and known platform limitations.
- Add scientific-method notes explaining assumptions, calibration ranges and appropriate interpretation.
- Document the boundary between MIT-licensed RUNA code and third-party software or data.
- Add a contribution guide and issue templates for reproducible bug reports.

## Estimated scope

The complete roadmap represents approximately **65–98 hours** of planned engineering, scientific validation, testing and documentation. A first implementation milestone will prioritise object bibliography and provenance, Bailer-Jones distances, data-source documentation, remote-service reliability and a foundational test suite.
