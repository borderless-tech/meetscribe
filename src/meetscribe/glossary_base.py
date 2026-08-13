"""Shipped base glossary: common tech/office vocabulary that general de+en dictionaries miss.

These are treated as *known* by the flagger, so they don't show up as broken-word false
positives (HubSpot, committen, Onboarding, …) drowning the human reviewer. Matched
case-insensitively (see :func:`lexicon.Lexicon`), so only one casing is needed — but German
anglicism *conjugations* are distinct tokens, so the frequent verbs list their common forms.

This is a starting point, merged with the user's persistent glossary + per-meeting names. Edit
freely; over-inclusion only means a genuinely-garbled instance of one of these won't be
flagged — an acceptable trade under human-in-the-loop review.
"""

from __future__ import annotations

# --- Companies / vendors ---------------------------------------------------------------
_COMPANIES = """
Microsoft Google Apple Amazon AWS Azure GCP Salesforce HubSpot SAP Oracle IBM Adobe
Atlassian Jira Confluence Slack Zoom Notion Figma Miro Canva Stripe Shopify GitHub GitLab
Bitbucket LinkedIn Meta Facebook Instagram WhatsApp Telegram Xing PwC KPMG Deloitte EY
Accenture McKinsey BCG Bain Cisco Dropbox Box Zapier Make Asana Trello Monday ClickUp
Zendesk Intercom Freshdesk Mailchimp Brevo Twilio SendGrid Datadog Sentry Snowflake
Databricks OpenAI Anthropic Nvidia Tesla Palantir Workday ServiceNow Pipedrive Airtable
Webex GoToMeeting Calendly DocuSign Miro Loom Grammarly Perplexity Mistral
"""

# --- Tools / products ------------------------------------------------------------------
_TOOLS = """
Teams Outlook Excel Word PowerPoint OneNote OneDrive SharePoint Sheets Docs Slides Gmail
Meet Drive Calendar Photoshop Illustrator InDesign Premiere Kubernetes Docker Terraform
Ansible Postman Insomnia VSCode IntelliJ Xcode Jenkins Grafana Kibana Elasticsearch Redis
Postgres MySQL MongoDB Kafka RabbitMQ Nginx Cloudflare Vercel Netlify Heroku
"""

# --- Business / office nouns (mostly anglicisms) ---------------------------------------
_NOUNS = """
Onboarding Offboarding Meeting Call Deadline Feature Roadmap Pipeline Lead Leads Stakeholder
Stakeholdern Workshop Slides Deck Follow-up Kickoff Standup Sprint Backlog Feature-Request
Feedback Update Review Alignment Mindset Benchmark Case Use-Case Insights Learnings
Deliverable Deliverables Scope Budget Timeline Milestone KPI KPIs OKR OKRs ROI CRM ERP SaaS
PaaS B2B B2C USP MVP POC Touchpoint Wording Setup Workflow Dashboard Template Webinar Webinare
Newsletter Outreach Sales Marketing Reporting Forecast Revenue Churn Funnel Conversion Growth
Traction Runway Cashflow Equity Cap-Table Due-Diligence Term-Sheet Valuation Bootstrapping
Scaleup Startup Startups Founder Co-Founder Founders Investor Investoren Pitch Pitches
Pitch-Deck Whitepaper Blogpost Podcast Roundtable Roundtables Webhook Webhooks API APIs
Dashboard Frontend Backend Fullstack DevOps Repository Repo Commit Merge-Request Pull-Request
Branch Release Deployment Feature-Flag Bug Bugfix Hotfix Ticket Tickets Epic Story Roadmaps
Persona Personas Segment Segmente Prospect Prospects Account Accounts Retention Acquisition
Referral Nurturing Sequencing Cadence Playbook Enablement Quota Commission Upsell Cross-Sell
"""

# --- Office-talk verbs (German anglicisms) — infinitive + frequent conjugations ---------
_VERBS = """
committen committe committen committet committed
brainstormen brainstorme brainstormt gebrainstormt
tracken tracke trackt getrackt
pitchen pitche pitcht gepitcht
onboarden onboarde onboardet onboarded
offboarden
deployen deploye deployt deployed
canceln cancele cancelt gecancelt gecancelled
updaten update updatet upgedatet geupdatet
downloaden downloade downgeloadet
uploaden uploade hochgeloadet
checken checke checkt gecheckt
matchen matche matcht gematcht
pushen pushe pusht gepusht
mergen merge mergt gemerged gemergt
sharen share sharet geshared
liken like likt geliked
posten poste postet gepostet
forwarden forwarde
mailen maile mailt gemailt
callen calle callt
googlen google googelt gegoogelt
managen manage managt gemanaged gemanagt
connecten connecte
presenten
reviewen reviewe reviewt
alignen aligne
scopen scope
framen frame
briefen briefe brieft gebrieft
debriefen
closen close clost geclosed
sourcen source
screenen screene
prioritisieren priorisieren priorisiere priorisiert
skypen skype
zoomen zoome
committen scrollen scrolle swipen liken teilen
"""


# --- Colloquial / spoken German the dictionary lacks (fillers, contractions, prefix verbs) -
# High-frequency in meetings; not in general dicts → otherwise a constant false-flag source.
_COLLOQUIAL = """
bzw naja joa sowas erstmal sozusagen quasi halt mega voll safe genau ne
reinspringen reinsteigen reinbekommen reinkloppen reinschauen reinhauen reingehen
runterarbeiten runterbrechen runtergebrochen runterfallen runterladen ranholen ranarbeiten
draufschauen durchsprechen durchgehen abstimmen nachhalten hochholen weiterdenken
"""


def _terms(*blocks: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for block in blocks:
        for tok in block.split():
            key = tok.lower()
            if key not in seen:
                seen.add(key)
                out.append(tok)
    return out


BASE_TERMS: list[str] = _terms(_COMPANIES, _TOOLS, _NOUNS, _VERBS, _COLLOQUIAL)
