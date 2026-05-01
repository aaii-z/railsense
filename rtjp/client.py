import os
from pathlib import Path
from typing import Optional

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass
from requests.auth import HTTPBasicAuth
from zeep import Client
from zeep.settings import Settings
from zeep.transports import Transport


def create_client(*, endpoint: Optional[str] = None) -> Client:
    wsdl_path = Path(__file__).resolve().parent / "jpservices.wsdl"
    if not wsdl_path.is_file():
        raise FileNotFoundError(f"WSDL not found at {wsdl_path}")
    wsdl_ref = wsdl_path.as_uri()

    user = os.environ.get("OJP_BASIC_AUTH_USER")
    password = os.environ.get("OJP_BASIC_AUTH_PASSWORD")
    if (user or password) and not (user and password):
        raise ValueError(
            "Set both OJP_BASIC_AUTH_USER and OJP_BASIC_AUTH_PASSWORD, or leave both unset."
        )
    if user and password:
        session = requests.Session()
        session.auth = HTTPBasicAuth(user, password)
        transport = Transport(session=session)
    else:
        transport = Transport()

    settings = Settings(
        strict=False,
        xsd_ignore_sequence_order=True,
    )
    client = Client(wsdl_ref, transport=transport, settings=settings)

    if endpoint is None:
        endpoint = os.environ.get("OJP_SOAP_ENDPOINT")

    if endpoint:
        client.service._binding_options["address"] = endpoint

    return client


def _print_operations() -> None:
    client = create_client()
    for service in client.wsdl.services.values():
        print(service.name)
        for port in service.ports.values():
            print(f"  port: {port.name} -> {port.binding_options.get('address')}")
            for op in port.binding._operations.values():
                print(f"    {op.name}")


if __name__ == "__main__":
    _print_operations()
