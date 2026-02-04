"""
This module defines all enum types used across the system
to represent fixed sets of domain-specific values such as
user roles, job work modes, employment types, and
application statuses.
"""

from enum import Enum

class UserRole(str, Enum):
    # defines roles of users in system
    ADMIN = "ADMIN"
    RECRUITER = "RECRUITER"
    CANDIDATE = "CANDIDATE"

class ModeOfWork(str, Enum):
    #  Represents modes of work
    ONSITE = "ONSITE"
    REMOTE = "REMOTE"
    HYBRID = "HYBRID"

class EmploymentType(str, Enum):
    # represents the type of employment
    FULL_TIME = "FULL_TIME"
    PART_TIME = "PART_TIME"
    INTERN = "INTERN"

class ApplicationStatus(str, Enum):
    # application status of users' sbmitted applications
    APPLIED = "APPLIED"
    UNDER_REVIEW = "UNDER_REVIEW"
    REJECTED = "REJECTED"
    ACCEPTED = "ACCEPTED"