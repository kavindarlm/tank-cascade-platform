# GEE service account for unattended runs

Interactive `earthengine authenticate` cannot run inside GitHub Actions
(no browser). Use a service account:

1. Google Cloud console -> project `gee-water-monitoring` -> IAM & Admin ->
   Service Accounts -> Create. Give it a name like `module1-daily`.
2. Create a JSON key for it, download the file.
3. Register the service account for Earth Engine at
   https://signup.earthengine.google.com/#!/service_accounts
   (and make sure it can read the asset `.../nachchaduwa-32-final`).
4. In the GitHub repo -> Settings -> Secrets and variables -> Actions, add:
   - `GEE_SA_EMAIL`  = the service account email
   - `GEE_KEY_JSON`  = the full contents of the JSON key file
5. The workflow writes the key to `gee_key.json` at run time, uses it, and
   deletes it before committing. The key is never stored in the repo.
