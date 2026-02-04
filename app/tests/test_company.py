from app.tests.factory import user_payload
import pytest
from uuid import uuid4
from app.core.enum import UserRole, ApplicationStatus, ModeOfWork, EmploymentType

# Test company creation for different user roles.
@pytest.mark.parametrize("role", [UserRole.RECRUITER, UserRole.ADMIN])
def test_create_company(client, auth_headers, role):
    identity=str(uuid4().hex)
    payload={
        "name": f"Company_{identity}",
        "description": f"Our company is the best",
        "website": f"{identity}.com",
        "location": "Gandhinagar",
        "domain": "Python",
        "company_size": 100
    }
    headers=auth_headers(role)
    response=client.post("/companies/", json=payload, headers=headers)
    assert response.status_code==201
    company_data=response.json()
    assert "id" in company_data
    assert "owner_id" in company_data

# Test fetching a company by ID.
@pytest.mark.parametrize("role", [UserRole.CANDIDATE, UserRole.RECRUITER, UserRole.ADMIN])
def test_get_company_by_id(client, get_created_company, auth_headers, role):
    headers=auth_headers(role)
    company_id = get_created_company["id"]
    response = client.get(f"/companies/{company_id}", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == company_id
    response = client.get("/companies/ssjk")
    assert response.status_code == 422

# Test updating company details.
def test_update_company(client, auth_headers, get_created_company):
    company_id = get_created_company["id"]
    headers = auth_headers(UserRole.ADMIN)

    payload = {
        "name": f"NEWNAME_{uuid4()}",
        "description": f"Our company is the best, new one",
        "website": f"{uuid4()}.com",
        "location": "Gandhinagar",
        "domain": "Python",
        "company_size": 100
    }
    response = client.put(f"/companies/{company_id}", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["description"] == "Our company is the best, new one"
    assert data["company_size"] == 100

# Test deleting a company.
def test_delete_company(client, auth_headers, get_created_company):
    headers = auth_headers(UserRole.ADMIN)
    company_id = get_created_company["id"]
    response = client.delete(f"/companies/{company_id}", headers=headers)
    assert response.status_code == 204
    response = client.get(f"/companies/{company_id}")
    assert response.status_code == 404

# Test listing all companies.
def test_list_companies(client):
    response = client.get("/companies/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1