from app.core.enum import UserRole, ApplicationStatus, ModeOfWork, EmploymentType

# Test job creation by a recruiter.
def test_create_job(client, auth_headers, job_payload, get_created_company):
    headers=auth_headers(role=UserRole.RECRUITER, current_organization=get_created_company["id"])
    response = client.post("/jobs/", json=job_payload, headers=headers)
    assert response.status_code == 201

# Test fetching a job by its ID.
def test_get_job_by_id(client, auth_headers, get_created_job):
    headers=auth_headers(role=UserRole.CANDIDATE)
    job_id=get_created_job["id"]
    response = client.get(f"/jobs/{job_id}", headers=headers)
    assert response.status_code==200

# Test job listing with pagination and filters.
def test_list_jobs(client, auth_headers, get_created_jobs_list):
    headers=auth_headers(role=UserRole.CANDIDATE)
    response=client.get("/jobs/?page=1&size=2", headers=headers)
    assert response.status_code==200
    response_data=response.json()
    assert "items" in response_data
    assert len(response_data["items"])==2
    assert "total" in response_data
    assert response_data["total"]>=3
    response=client.get("/jobs/?search_query=inter", headers=headers)
    assert response.status_code == 200
    response_data = response.json()
    assert len(response_data["items"]) >= 2
    response=client.get("/jobs/?employment_type=FULL_TIME", headers=headers)
    assert response.status_code == 200
    response_data = response.json()
    assert len(response_data["items"]) == 1
    response=client.get("/jobs/?tags=ONSITE", headers=headers)
    assert response.status_code == 200
    response_data = response.json()
    assert len(response_data["items"]) == 2
    for item in response_data["items"]:
        assert "ONSITE" in item["tags"]

# Test updating an existing job.
def test_update_job(client, auth_headers, job_payload, get_created_company, get_created_job):
    company_id=get_created_company["id"]
    headers=auth_headers(role=UserRole.RECRUITER, current_organization=company_id)
    job_id=get_created_job["id"]
    payload=job_payload
    payload["location"]="San Francisco"
    response=client.put(f"/jobs/{job_id}", json=payload, headers=headers)
    assert response.status_code==200
    response_data=response.json()
    assert response_data["location"]=="San Francisco"

# Test deleting a job.
def test_delete_job(client, auth_headers, get_created_company, get_created_job):
    company_id=get_created_company["id"]
    headers=auth_headers(role=UserRole.RECRUITER, current_organization=company_id)
    job_id=get_created_job["id"]
    response=client.delete(f"/jobs/{job_id}", headers=headers)
    assert response.status_code==204