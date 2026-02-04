"""
Tests for application-related API endpoints.

These tests cover:
- Creating job applications
- Retrieving applications
- Updating application status
- Deleting applications
"""

from app.core.enum import ApplicationStatus, UserRole, EmploymentType, ModeOfWork

# Test that a candidate can successfully apply for a job.
def test_create_application(client, auth_headers, application_payload, get_created_job, temp_upload_dir):
    job_id=get_created_job["id"]
    headers=auth_headers(UserRole.CANDIDATE)
    content=b"testing resume content"
    response=client.post(f"/applications/jobs/{job_id}/apply", headers=headers, data=application_payload, files={"resume":("test_resume.pdf", content, "application/pdf")})
    assert response.status_code==201

# Test retrieving applications by ID, job, and user.
def test_get_application(client, auth_headers, get_created_job, get_created_application):
    job_id=get_created_job["id"]
    company_id=get_created_job["company_id"]
    headers=auth_headers(role=UserRole.RECRUITER, current_organization=company_id)
    application_id=get_created_application["id"]
    response=client.get(f"/applications/{application_id}", headers=headers)
    assert response.status_code==200
    response=client.get(f"/applications/jobs/{job_id}", headers=headers)
    assert response.status_code==200
    user_reponse=client.get("/users/me", headers=headers)
    assert user_reponse.status_code == 200
    data = user_reponse.json()
    assert "id" in data
    user_id=data["id"]
    response=client.get(f"/applications/users/{user_id}", headers=headers)
    assert response.status_code==200

# Test that a recruiter can update application status.
def test_application_status_update(client, auth_headers, get_created_application):
    headers=auth_headers(role=UserRole.RECRUITER)
    application_id=get_created_application["id"]
    response=client.put(f"/applications/{application_id}/?new_status=APPLIED", headers=headers)
    assert response.status_code==200

# Test that a recruiter can delete an application.
def test_delete_application(client, auth_headers, get_created_application):
    headers=auth_headers(role=UserRole.RECRUITER)
    application_id=get_created_application["id"]
    response=client.delete(f"/applications/{application_id}", headers=headers)
    assert response.status_code==204
