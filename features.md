# Phishing Dataset Features

## Overview

This dataset contains features extracted from URLs to identify phishing websites. The dataset includes 235,795 URLs, with 57.2% being phishing attempts and 42.8% legitimate sites.

## Feature Categories

### Basic URL Information

- **URL/Domain**: Website URL and domain name
- **URLLength/DomainLength**: Length metrics (longer URLs often indicate phishing)
- **IsDomainIP**: Boolean indicating if domain is an IP address
- **TLD**: Top-Level Domain (com, org, net)
- **TLDLegitimateProb**: Historical probability of TLD legitimacy
- **NoOfSubDomain**: Number of subdomains present

### URL Composition Features

- **URLSimilarityIndex**: Similarity score with known legitimate domains
- **CharContinuationRate**: Character repetition frequency
- **HasObfuscation**: Presence of obfuscated characters
- **NoOfObfuscatedChar**: Count of obfuscated characters
- **ObfuscationRatio**: Ratio of obfuscated characters
- **Letter/DigitRatioInURL**: Character type distribution
- **NoOfEquals/QMark/AmpersandInURL**: Query parameter indicators
- **SpecialCharRatioInURL**: Special character frequency

### Security Indicators

- **IsHTTPS**: HTTPS protocol usage
- **Robots**: Presence of robots.txt
- **HasFavicon**: Website favicon presence

### Page Content Features

- **LineOfCode**: Total lines of HTML code
- **LargestLineLength**: Maximum line length
- **HasTitle**: Title tag presence
- **Title**: Page title content
- **HasDescription**: Meta description tag presence
- **DomainTitleMatchScore**: Domain-title similarity
- **URLTitleMatchScore**: URL-title similarity
- **HasCopyrightInfo**: Presence of copyright information (legitimate sites more likely to have this)
- **HasSocialNet**: Presence of social network links (legitimate sites more likely to have these)

### Behavioral Features

- **NoOfURLRedirect**: Number of redirects
- **NoOfSelfRedirect**: Self-redirect count
- **HasExternalFormSubmit**: External form submission
- **HasHiddenFields**: Hidden form fields
- **HasPasswordField**: Password field presence
