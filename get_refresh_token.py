from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def get_refresh_token():
    flow = InstalledAppFlow.from_client_secrets_file('client_secret_321468335432-nfb1br0r9qc31p4eckq1g2failqalcpv.apps.googleusercontent.com.json', SCOPES)
    credentials = flow.run_local_server(port=0)
    print(f"Refresh Token: {credentials.refresh_token}")
    print(f"Client ID: {credentials.client_id}")
    print(f"Client Secret: {credentials.client_secret}")

if __name__ == '__main__':
    get_refresh_token() 