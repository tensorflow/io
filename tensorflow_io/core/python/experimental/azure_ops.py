# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""tensorflow-io azure file system import"""


import tensorflow_io.core.python.ops  # pylint: disable=unused-import


def authenticate_with_device_code(account_name):
    """Setup storage tokens by authenticating with device code
    and use management APIs.

    Args:
        account_name (str): The storage account name for which to authenticate
    """

    import urllib  # pylint: disable=import-outside-toplevel
    import json  # pylint: disable=import-outside-toplevel
    import os  # pylint: disable=import-outside-toplevel
    from tensorflow.python.platform import (  # pylint: disable=import-outside-toplevel
        tf_logging as log,
    )

    try:
        from adal import (  # pylint: disable=import-outside-toplevel
            AuthenticationContext,
        )
    except ModuleNotFoundError:
        log.error(
            "Please install adal library with `python -m pip install -U adal`"
            "to use the device code authentication method"
        )
        return

    ctx = AuthenticationContext("https://login.microsoftonline.com/common")

    storage_resource = "https://management.azure.com/"
    # Current multi-tenant client registerd in my AzureAD tenant
    client_id = "8c375311-7f4c-406c-84f8-03dfe11ba2d3"

    device_code = ctx.acquire_user_code(resource=storage_resource, client_id=client_id)

    # Display authentication message to user to action in their browser
    log.warn(device_code["message"])

    token_response = ctx.acquire_token_with_device_code(
        resource=storage_resource, user_code_info=device_code, client_id=client_id
    )

    headers = {"Authorization": "Bearer " + token_response["accessToken"]}

    subscription_list_req = urllib.request.Request(
        url="https://management.azure.com/subscriptions?api-version=2016-06-01",
        headers=headers,
    )

    with urllib.request.urlopen(subscription_list_req) as f:
        subscriptions = json.load(f)
    subscriptions = subscriptions["value"]

    storage_account = None
    for subscription in subscriptions:
        url = "https://management.azure.com/subscriptions/{}/providers/Microsoft.Storage/storageAccounts?api-version=2019-04-01".format(
            subscription["subscriptionId"]
        )
        storage_account_list_req = urllib.request.Request(url=url, headers=headers)

        with urllib.request.urlopen(storage_account_list_req) as f:
            storage_accounts = json.load(f)

        storage_accounts = storage_accounts["value"]
        account_by_name = [s for s in storage_accounts if s.get("name") == account_name]
        if any(account_by_name):
            storage_account = account_by_name[0]
            break

    if storage_account is None:
        log.error(
            "Couldn't find storage account {} in any "
            "available subscription".format(account_name)
        )
        return

    url = "https://management.azure.com/{}/listKeys?api-version=2019-04-01".format(
        storage_account["id"]
    )
    storage_list_keys_req = urllib.request.Request(
        url=url, headers=headers, method="POST"
    )

    with urllib.request.urlopen(storage_list_keys_req) as f:
        account_keys = json.load(f)

    os.environ["TF_AZURE_STORAGE_KEY"] = account_keys["keys"][0]["value"]
    log.info(
        "Successfully set account key environment for {} "
        "storage account".format(account_name)
    )
