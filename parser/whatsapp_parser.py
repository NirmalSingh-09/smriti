import re
import pandas as pd

def parse_whatsapp_chat(file_path, target_person):

    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}\s?(?:\u202f)?(?:am|pm|AM|PM)?)\s-\s([^:]+):\s(.+)'

    messages = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    skip_phrases = [
        '<Media omitted>', 'This message was deleted',
        'You deleted this message', '<This message was edited>',
        'this message was edited', 'message was edited',
        'Messages and calls are end-to-end encrypted'
    ]

    for line in lines:
        line = line.strip()
        match = re.match(pattern, line)
        if match:
            date_str, time_str, sender, message = match.groups()
            sender = sender.strip()

            if any(phrase.lower() in message.lower() for phrase in skip_phrases):
                continue

            messages.append({
                'date': date_str,
                'time': time_str,
                'sender': sender,
                'message': message.strip()
            })

    df = pd.DataFrame(messages)

    if df.empty:
        print("No messages parsed. Check the chat format.")
        return df

    target_df = df[df['sender'].str.lower() == target_person.lower()].copy()
    target_df.reset_index(drop=True, inplace=True)

    print(f"Total messages found: {len(df)}")
    print(f"Messages from '{target_person}': {len(target_df)}")

    return target_df