# PCI-DSS Compliance Standard

## Scope

This standard describes how NovaBank protects payment card data in line with
the Payment Card Industry Data Security Standard (PCI-DSS). It applies to any
system that stores, processes, or transmits cardholder data.

## Cardholder Data Protection

Cardholder data must be **encrypted both at rest and in transit** using
strong cryptography. The card verification value (**CVV/CVC is never
stored**) after authorization, under any circumstance. The Primary Account
Number (PAN) is masked when displayed, showing at most the first six and last
four digits.

## Network Segmentation

Systems handling cardholder data are isolated in a dedicated network segment,
separated from the corporate network by firewalls. Access between segments is
denied by default and allowed only where a documented business need exists.

## Vulnerability Management

Internal and external **vulnerability scans are performed quarterly**, and
after any significant change. Critical vulnerabilities must be remediated
within 30 days. Penetration testing is conducted at least annually.

## Access Control

Access to cardholder data is granted strictly on a **need-to-know** basis.
Every user has a unique ID, and multi-factor authentication is required for
all administrative access. All access to cardholder data is logged.
