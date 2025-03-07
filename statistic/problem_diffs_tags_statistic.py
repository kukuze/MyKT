# SQL 查询
from statistic.mysql import connect_to_database, execute_query


def process_raw_data(results):
    raw_data = []
    for row in results:
        tag_name, difficulty, count = row
        # For each result, map it into the required format [tag_name, difficulty, count]
        raw_data.append([tag_name, difficulty, count])
    return raw_data
query = """
SELECT 
        pt.tag_name, 
        cp.difficulty,
        COUNT(cpt.codeforces_problem_id) AS count
    FROM 
        problem_tag pt
    JOIN 
        codeforces_problems_tag_map cpt ON pt.id = cpt.tag_id
    JOIN 
        codeforces_problems cp ON cp.id = cpt.codeforces_problem_id
    WHERE 
        cp.difficulty != 0
    GROUP BY 
        pt.tag_name, cp.difficulty
"""

# Main execution
if __name__ == "__main__":
    conn, cursor = connect_to_database()
    results = execute_query(cursor, query, None)
    raw_data = process_raw_data(results)

    # Print the raw_data to verify the output
    print(raw_data)

    # Step 5: Clean up (close the connection and cursor)
    cursor.close()
    conn.close()
