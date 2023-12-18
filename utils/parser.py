class QueryParser():
  def __init__(self, paper_indicator='$'):
    self.paper_indicator = paper_indicator

  def is_valid_query(self, initial_query):

    valid = False

    if initial_query.count(self.paper_indicator) % 2 != 0:
      return valid

    valid = True
    return valid

  def parse_query(self, inital_query):
    response = {
        'target_papers': [],
        'query': [],
        'alert_msg': 'Wait for answers!',
    }
    if not self.is_valid_query(inital_query):
      response['alert_msg'] = 'please valid input'
      return response # Not valid case

    chunks = inital_query.split('$')
    response['target_papers'] = [chunk for idx,chunk in enumerate(chunks) if idx%2==1 and len(chunk)>0]
    query = [chunk for idx,chunk in enumerate(chunks) if idx%2!=1]
    query = ' '.join(query).replace(', ','').replace('  ', '')
    if query[0] == ' ':
      query = query[1:]
    response['query'] = query

    return response
